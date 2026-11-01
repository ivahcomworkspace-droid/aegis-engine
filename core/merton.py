"""
Aegis Structural Merton Option Engine
Inverts latent enterprise assets (V_A) and asset volatility (sigma_A) from observable
equity market cap (E), equity volatility (sigma_E), and default barrier (D = STD + 0.5 * LTD).
Solves the coupled non-linear system via SciPy's modified Powell hybrid method ('hybr')
with an exact analytical Jacobian matrix for sub-2ms convergence.
"""

import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import root

try:
    from aegis_engine.schemas.credit_models import MertonInput, MertonOutput
except ModuleNotFoundError:
    from schemas.credit_models import MertonInput, MertonOutput


class MertonSolver:
    """
    High-performance, institutional-grade Merton structural credit solver.
    """

    @staticmethod
    def _system_and_jacobian(x: np.ndarray, E: float, sigma_E: float, D: float, r: float, T: float):
        """
        Evaluates system F(x) and exact analytical Jacobian matrix J(x).
        x = [V_A, sigma_A]
        """
        V_A, sigma_A = x[0], x[1]

        # Guard against non-positive evaluations in solver iterations
        if V_A <= 1e-6:
            V_A = 1e-6
        if sigma_A <= 1e-6:
            sigma_A = 1e-6

        sqrt_T = np.sqrt(T)
        sigma_A_sqrt_T = sigma_A * sqrt_T
        log_VA_D = np.log(V_A / D)

        d1 = (log_VA_D + (r + 0.5 * sigma_A**2) * T) / sigma_A_sqrt_T
        d2 = d1 - sigma_A_sqrt_T

        Phi_d1 = norm.cdf(d1)
        Phi_d2 = norm.cdf(d2)
        phi_d1 = norm.pdf(d1)

        # Coupled Merton non-linear equations
        # F1 = V_A * Phi(d1) - D * exp(-r*T) * Phi(d2) - E
        # F2 = (V_A / E) * Phi(d1) * sigma_A - sigma_E
        f1 = V_A * Phi_d1 - D * np.exp(-r * T) * Phi_d2 - E
        f2 = (V_A / E) * Phi_d1 * sigma_A - sigma_E

        # Exact Analytical Jacobian entries
        # dF1 / dV_A = Phi(d1) (due to Black-Scholes delta identity V_A*phi(d1) - D*exp(-rT)*phi(d2) = 0)
        # dF1 / dsigma_A = V_A * sqrt(T) * phi(d1) (vega)
        # dF2 / dV_A = (sigma_A / E) * [Phi(d1) + phi(d1) / (sigma_A * sqrt(T))]
        # dF2 / dsigma_A = (V_A / E) * [Phi(d1) + phi(d1) * ((r + 0.5*sigma_A^2)/sigma_A^2 - ln(V_A/D)/(sigma_A^2 * T)) * sqrt(T)]
        dF1_dVA = Phi_d1
        dF1_dsigmaA = V_A * sqrt_T * phi_d1

        dF2_dVA = (sigma_A / E) * (Phi_d1 + phi_d1 / sigma_A_sqrt_T)

        d_d1_dsigmaA = - (log_VA_D + r * T) / (sigma_A**2 * sqrt_T) + 0.5 * sqrt_T
        dF2_dsigmaA = (V_A / E) * (Phi_d1 + sigma_A * phi_d1 * d_d1_dsigmaA)

        J = np.array([
            [dF1_dVA, dF1_dsigmaA],
            [dF2_dVA, dF2_dsigmaA]
        ], dtype=np.float64)

        return np.array([f1, f2], dtype=np.float64), J

    @classmethod
    def solve(cls, merton_in: MertonInput) -> MertonOutput:
        """
        Solves the Merton model for latent enterprise assets and volatility.
        Includes asymptotic clamp for zero debt (D = 0) and analytical Jacobian.
        """
        start_time = time.perf_counter()

        E = float(merton_in.equity_value)
        sigma_E = float(merton_in.equity_volatility)
        D = float(merton_in.default_barrier)
        r = float(merton_in.risk_free_rate)
        mu_A = float(merton_in.drift_mu) if merton_in.drift_mu is not None else r
        T = float(merton_in.maturity)
        lgd = float(merton_in.loss_given_default)

        # Asymptotic Clamp: D = 0 => V_A = E, sigma_A = sigma_E, DD = 99.0, PD = 0.0
        if D <= 1e-9:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return MertonOutput(
                asset_value=E,
                asset_volatility=sigma_E,
                default_barrier=0.0,
                d1=99.0,
                d2=99.0,
                distance_to_default_physical=99.0,
                default_probability_physical=0.0,
                distance_to_default_risk_neutral=99.0,
                default_probability_risk_neutral=0.0,
                credit_spread_bps=0.0,
                solver_converged=True,
                solver_message="Asymptotic clamp: Zero debt (D=0)",
                solver_time_ms=round(elapsed_ms, 3),
                leverage_ratio=0.0
            )

        # Initial parameter estimates
        # Lower bound for V_A is E + D*exp(-r*T)
        V_A_init = E + D * np.exp(-r * T)
        # By Merton relation sigma_A ~ sigma_E * (E / V_A)
        sigma_A_init = max(sigma_E * (E / V_A_init), 0.01)

        x0 = np.array([V_A_init, sigma_A_init], dtype=np.float64)

        def func_system(x):
            f, _ = cls._system_and_jacobian(x, E, sigma_E, D, r, T)
            return f

        def jac_system(x):
            _, J = cls._system_and_jacobian(x, E, sigma_E, D, r, T)
            return J

        # SciPy Powell hybrid method ('hybr') with analytical Jacobian
        res = root(
            fun=func_system,
            x0=x0,
            method='hybr',
            jac=jac_system,
            options={'xtol': 1e-9, 'maxfev': 500}
        )

        converged = bool(res.success)
        solver_msg = str(res.message)
        V_A, sigma_A = float(res.x[0]), float(res.x[1])

        # Numerical fallback if hybr hit edge of domain
        if not converged or V_A <= 0 or sigma_A <= 0 or np.isnan(V_A) or np.isnan(sigma_A):
            # Fallback to bounded least squares / LM
            from scipy.optimize import least_squares
            res_fallback = least_squares(
                fun=func_system,
                x0=x0,
                jac=jac_system,
                bounds=([E * 0.5, 0.001], [np.inf, 5.0]),
                ftol=1e-8,
                xtol=1e-8
            )
            converged = bool(res_fallback.success)
            solver_msg = f"Fallback least_squares: {res_fallback.message}"
            V_A, sigma_A = float(res_fallback.x[0]), float(res_fallback.x[1])

        # Enforce physical positivity
        V_A = max(V_A, E)
        sigma_A = max(sigma_A, 0.0001)

        sqrt_T = np.sqrt(T)
        sigma_A_sqrt_T = sigma_A * sqrt_T
        log_VA_D = np.log(V_A / D)

        # d1 and d2 under risk-neutral measure
        d1 = (log_VA_D + (r + 0.5 * sigma_A**2) * T) / sigma_A_sqrt_T
        d2 = d1 - sigma_A_sqrt_T

        # Physical Distance to Default (using physical drift mu_A)
        # DD = [ln(V_A / D) + (mu_A - 0.5*sigma_A^2)*T] / (sigma_A * sqrt(T))
        dd_physical = (log_VA_D + (mu_A - 0.5 * sigma_A**2) * T) / sigma_A_sqrt_T
        pd_physical = float(norm.cdf(-dd_physical))

        # Risk-Neutral Distance to Default & Implied Credit Spread
        # DD_RN = [ln(V_A / D) + (r - 0.5*sigma_A^2)*T] / (sigma_A * sqrt(T)) = d2
        dd_rn = (log_VA_D + (r - 0.5 * sigma_A**2) * T) / sigma_A_sqrt_T
        pd_rn = float(norm.cdf(-dd_rn))

        # Risk-neutral theoretical credit spread
        # Spread (bps) = -ln(1 - PD_RN * LGD) / T * 10,000
        pd_loss = min(pd_rn * lgd, 0.999999)
        spread_bps = float(-np.log(1.0 - pd_loss) / T * 10000.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return MertonOutput(
            asset_value=round(V_A, 4),
            asset_volatility=round(sigma_A, 6),
            default_barrier=round(D, 4),
            d1=round(float(d1), 6),
            d2=round(float(d2), 6),
            distance_to_default_physical=round(float(dd_physical), 4),
            default_probability_physical=round(pd_physical, 6),
            distance_to_default_risk_neutral=round(float(dd_rn), 4),
            default_probability_risk_neutral=round(pd_rn, 6),
            credit_spread_bps=round(spread_bps, 2),
            solver_converged=converged,
            solver_message=solver_msg,
            solver_time_ms=round(elapsed_ms, 3),
            leverage_ratio=round(D / V_A, 4)
        )


def solve_merton(merton_in: MertonInput) -> MertonOutput:
    """Convenience wrapper for Merton solver."""
    return MertonSolver.solve(merton_in)
