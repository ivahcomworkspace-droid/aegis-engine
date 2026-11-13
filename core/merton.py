"""
Aegis Structural Merton Option Engine
Bivariate Non-Linear Solver with Closed-Form Analytical Jacobian Matrix.
Inverts latent enterprise assets (V_A) and asset volatility (sigma_A) from observable
equity market cap (E), equity volatility (sigma_E), and default barrier (D = STD + 0.5 * LTD).
Achieves sub-2ms convergence via SciPy's modified Powell hybrid method ('hybr').
"""

import time
from typing import Dict, Tuple, Any
import numpy as np
import scipy.stats as stats
from scipy.optimize import root

try:
    from aegis_engine.schemas.credit_models import MertonInput, MertonOutput
except ModuleNotFoundError:
    from schemas.credit_models import MertonInput, MertonOutput


def merton_objective_and_jacobian(
    vars_vec: np.ndarray,
    E: float,
    sigma_E: float,
    D: float,
    r: float,
    T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the bivariate coupled Merton non-linear system and its exact closed-form analytical Jacobian.
    vars_vec = [V_A, sigma_A]
    """
    v_a, sigma_a = vars_vec[0], vars_vec[1]
    
    # Asymptotic domain clamp
    if v_a <= 0.0 or sigma_a <= 1e-6:
        return np.array([1e8, 1e8], dtype=np.float64), np.eye(2, dtype=np.float64) * 1e8

    sqrt_T = np.sqrt(T)
    d1 = (np.log(v_a / D) + (r + 0.5 * sigma_a**2) * T) / (sigma_a * sqrt_T)
    d2 = d1 - sigma_a * sqrt_T

    phi_d1 = stats.norm.pdf(d1)
    Phi_d1 = stats.norm.cdf(d1)
    Phi_d2 = stats.norm.cdf(d2)

    # Coupled Merton Non-Linear Residuals:
    # F1 = V_A * Phi(d1) - D * exp(-rT) * Phi(d2) - E = 0
    # F2 = (V_A / E) * Phi(d1) * sigma_A - sigma_E = 0
    f1 = v_a * Phi_d1 - D * np.exp(-r * T) * Phi_d2 - E
    f2 = (v_a / E) * Phi_d1 * sigma_a - sigma_E
    residuals = np.array([f1, f2], dtype=np.float64)

    # Exact Analytical Jacobian Matrix Entries:
    # dF1 / dV_A = Phi(d1)
    # dF1 / dsigma_A = V_A * sqrt(T) * phi(d1)
    # dF2 / dV_A = (sigma_A / E) * [ Phi(d1) + phi(d1) / (sigma_A * sqrt(T)) ]
    # dF2 / dsigma_A = (V_A / E) * [ Phi(d1) + sigma_A * phi(d1) * d(d1)/d(sigma_A) ]
    df1_dVa = Phi_d1
    df1_dsigmaA = v_a * sqrt_T * phi_d1

    df2_dVa = (sigma_a / E) * (Phi_d1 + phi_d1 / (sigma_a * sqrt_T))

    d_d1_dsigmaA = - (np.log(v_a / D) + r * T) / (sigma_a**2 * sqrt_T) + 0.5 * sqrt_T
    df2_dsigmaA = (v_a / E) * (Phi_d1 + sigma_a * phi_d1 * d_d1_dsigmaA)

    jacobian = np.array([
        [df1_dVa, df1_dsigmaA],
        [df2_dVa, df2_dsigmaA]
    ], dtype=np.float64)

    return residuals, jacobian


def solve_structural_merton(
    equity_market_cap: float,
    equity_volatility: float,
    short_term_debt: float,
    long_term_debt: float,
    risk_free_rate: float,
    asset_drift_mu: float,
    maturity_years: float = 1.0,
    recovery_rate: float = 0.40
) -> Dict[str, Any]:
    """
    High-performance institutional solver for latent enterprise assets and default risk.

    Returns dictionary containing:
    - asset_value: Inverted enterprise asset value V_A
    - asset_volatility: Inverted enterprise asset volatility sigma_A
    - distance_to_default: Physical Distance-to-Default (DD) under asset drift mu
    - physical_default_prob: Physical Probability of Default Phi(-DD)
    - risk_neutral_default_prob: Risk-neutral Probability of Default Phi(-d2_rn)
    - implied_credit_spread_bps: Constant-recovery approximate credit spread in bps (-ln(1 - PD*LGD) / T * 10000)
    - structural_merton_spread_bps: Exact structural Merton credit spread in bps (-1/T * ln(Phi(d2) + (V_A*e^(rT)/D)*Phi(-d1)) * 10000)
    - leverage_ratio: Default barrier to Asset value ratio (D / V_A)
    - default_barrier: Merton default barrier D = STD + 0.5 * LTD
    - solver_converged: True if residual_norm < 1e-4 and V_A >= E - 1e-6
    - solver_used: Active solver employed ('hybr', 'trf_fallback', or 'zero_debt_exact')
    - residual_norm: L2 norm of dimensionless system residuals
    """
    E = float(equity_market_cap)
    sigma_E = float(equity_volatility)
    D = float(short_term_debt + 0.5 * long_term_debt)
    r = float(risk_free_rate)
    mu = float(asset_drift_mu)
    T = float(maturity_years)
    sqrt_T = np.sqrt(T)

    # Zero-Debt Asymptotic Guard
    if D <= 0.0:
        return {
            "asset_value": E,
            "asset_volatility": sigma_E,
            "distance_to_default": 99.0,
            "physical_default_prob": 0.0,
            "risk_neutral_default_prob": 0.0,
            "implied_credit_spread_bps": 0.0,
            "structural_merton_spread_bps": 0.0,
            "leverage_ratio": 0.0,
            "default_barrier": 0.0,
            "solver_converged": True,
            "solver_used": "zero_debt_exact",
            "residual_norm": 0.0,
        }

    # Dynamic starting heuristics
    v_a_init = E + D
    sigma_a_init = max(0.01, sigma_E * (E / v_a_init))

    sol = root(
        fun=lambda x: merton_objective_and_jacobian(x, E, sigma_E, D, r, T)[0],
        x0=np.array([v_a_init, sigma_a_init]),
        jac=lambda x: merton_objective_and_jacobian(x, E, sigma_E, D, r, T)[1],
        method='hybr',
        options={'xtol': 1e-9}
    )

    solver_used = "hybr"
    sol_success = bool(sol.success)
    solver_message = str(sol.message)

    if not sol.success or sol.x[0] < E - 1e-6 or sol.x[1] <= 0:
        # Fallback to bounded least squares if hybrid solver encounters numerical edge
        from scipy.optimize import least_squares
        res_fb = least_squares(
            fun=lambda x: merton_objective_and_jacobian(x, E, sigma_E, D, r, T)[0],
            x0=np.array([v_a_init, sigma_a_init]),
            jac=lambda x: merton_objective_and_jacobian(x, E, sigma_E, D, r, T)[1],
            bounds=([E, 0.001], [np.inf, 5.0]),
            xtol=1e-8,
            ftol=1e-8
        )
        solver_used = "trf_fallback"
        sol_success = bool(res_fb.success)
        solver_message = str(res_fb.message)
        v_a, sigma_a = float(res_fb.x[0]), float(res_fb.x[1])
    else:
        v_a, sigma_a = float(sol.x[0]), float(sol.x[1])

    # Recompute dimensionless equation residuals res_vec at (v_a, sigma_a)
    d1_calc = (np.log(v_a / D) + (r + 0.5 * sigma_a**2) * T) / (sigma_a * sqrt_T)
    d2_calc = d1_calc - sigma_a * sqrt_T
    phi_d1_calc = stats.norm.cdf(d1_calc)
    phi_d2_calc = stats.norm.cdf(d2_calc)

    res_vec = np.array([
        (v_a * phi_d1_calc - D * np.exp(-r * T) * phi_d2_calc - E) / E,
        (v_a / E) * phi_d1_calc * sigma_a - sigma_E
    ], dtype=np.float64)
    residual_norm = float(np.linalg.norm(res_vec))

    # If hybr gave residual_norm >= 1e-4, try trf_fallback before declaring failure
    if solver_used == "hybr" and (residual_norm >= 1e-4 or v_a < E - 1e-6):
        from scipy.optimize import least_squares
        res_fb = least_squares(
            fun=lambda x: merton_objective_and_jacobian(x, E, sigma_E, D, r, T)[0],
            x0=np.array([v_a_init, sigma_a_init]),
            jac=lambda x: merton_objective_and_jacobian(x, E, sigma_E, D, r, T)[1],
            bounds=([E, 0.001], [np.inf, 5.0]),
            xtol=1e-8,
            ftol=1e-8
        )
        solver_used = "trf_fallback"
        sol_success = bool(res_fb.success)
        solver_message = str(res_fb.message)
        v_a, sigma_a = float(res_fb.x[0]), float(res_fb.x[1])

        d1_calc = (np.log(v_a / D) + (r + 0.5 * sigma_a**2) * T) / (sigma_a * sqrt_T)
        d2_calc = d1_calc - sigma_a * sqrt_T
        phi_d1_calc = stats.norm.cdf(d1_calc)
        phi_d2_calc = stats.norm.cdf(d2_calc)

        res_vec = np.array([
            (v_a * phi_d1_calc - D * np.exp(-r * T) * phi_d2_calc - E) / E,
            (v_a / E) * phi_d1_calc * sigma_a - sigma_E
        ], dtype=np.float64)
        residual_norm = float(np.linalg.norm(res_vec))

    solver_converged = bool(sol_success and residual_norm < 1e-4 and v_a >= E - 1e-6)
    if not solver_converged:
        raise ArithmeticError(
            f"Merton solver ({solver_used}) failed to converge: {solver_message} (residual_norm={residual_norm:.2e})"
        )

    # Physical Distance-to-Default and default probability under drift mu
    dd_physical = (np.log(v_a / D) + (mu - 0.5 * sigma_a**2) * T) / (sigma_a * sqrt_T)
    pd_physical = float(stats.norm.cdf(-dd_physical))

    # Risk-neutral d1, d2, and default probability
    d1_rn = (np.log(v_a / D) + (r + 0.5 * sigma_a**2) * T) / (sigma_a * sqrt_T)
    d2_rn = d1_rn - sigma_a * sqrt_T
    pd_rn = float(stats.norm.cdf(-d2_rn))

    # Fix Me3: Exact structural Merton credit spread (bps) per Merton (1974)
    structural_merton_spread_bps = float(
        -(1.0 / T) * np.log(np.clip(stats.norm.cdf(d2_rn) + (v_a * np.exp(r * T) / D) * stats.norm.cdf(-d1_rn), 1e-15, 1.0)) * 10000.0
    ) if D > 0.0 else 0.0

    # Constant-recovery approximate spread (bps) retained for backwards compatibility
    lgd = 1.0 - recovery_rate
    approx_spread_bps = float(- (np.log(max(1e-7, 1.0 - min(pd_rn * lgd, 0.999999))) / T) * 10000.0)

    return {
        "asset_value": v_a,
        "asset_volatility": sigma_a,
        "distance_to_default": float(dd_physical),
        "physical_default_prob": pd_physical,
        "risk_neutral_default_prob": pd_rn,
        "implied_credit_spread_bps": approx_spread_bps,
        "structural_merton_spread_bps": structural_merton_spread_bps,
        "leverage_ratio": float(D / v_a),
        "default_barrier": D,
        "solver_converged": solver_converged,
        "solver_used": solver_used,
        "residual_norm": residual_norm,
    }


class MertonSolver:
    """
    High-performance class wrapper for backward-compatibility with Pydantic contracts and test suites.
    """
    _system_and_jacobian = staticmethod(merton_objective_and_jacobian)

    @classmethod
    def solve(cls, merton_in: MertonInput) -> MertonOutput:
        start_time = time.perf_counter()
        D = float(merton_in.default_barrier)

        if D <= 1e-9:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return MertonOutput(
                asset_value=float(merton_in.equity_value),
                asset_volatility=float(merton_in.equity_volatility),
                default_barrier=0.0,
                d1=99.0,
                d2=99.0,
                distance_to_default_physical=99.0,
                default_probability_physical=0.0,
                distance_to_default_risk_neutral=99.0,
                default_probability_risk_neutral=0.0,
                credit_spread_bps=0.0,
                structural_merton_spread_bps=0.0,
                solver_converged=True,
                solver_used="zero_debt_exact",
                residual_norm=0.0,
                solver_message="Asymptotic clamp: Zero debt (D=0)",
                solver_time_ms=round(elapsed_ms, 3),
                leverage_ratio=0.0
            )

        recovery_rate = 1.0 - float(merton_in.loss_given_default)
        asset_drift = float(merton_in.drift_mu) if merton_in.drift_mu is not None else float(merton_in.risk_free_rate)
        
        res = solve_structural_merton(
            equity_market_cap=float(merton_in.equity_value),
            equity_volatility=float(merton_in.equity_volatility),
            short_term_debt=float(merton_in.short_term_debt),
            long_term_debt=float(merton_in.long_term_debt),
            risk_free_rate=float(merton_in.risk_free_rate),
            asset_drift_mu=asset_drift,
            maturity_years=float(merton_in.maturity),
            recovery_rate=recovery_rate
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        T = float(merton_in.maturity)
        sqrt_T = np.sqrt(T)

        d1 = (np.log(res["asset_value"] / D) + (float(merton_in.risk_free_rate) + 0.5 * res["asset_volatility"]**2) * T) / (res["asset_volatility"] * sqrt_T)
        d2 = d1 - res["asset_volatility"] * sqrt_T
        dd_rn = (np.log(res["asset_value"] / D) + (float(merton_in.risk_free_rate) - 0.5 * res["asset_volatility"]**2) * T) / (res["asset_volatility"] * sqrt_T)

        return MertonOutput(
            asset_value=round(res["asset_value"], 4),
            asset_volatility=round(res["asset_volatility"], 6),
            default_barrier=round(D, 4),
            d1=round(float(d1), 6),
            d2=round(float(d2), 6),
            distance_to_default_physical=round(res["distance_to_default"], 4),
            default_probability_physical=round(res["physical_default_prob"], 6),
            distance_to_default_risk_neutral=round(float(dd_rn), 4),
            default_probability_risk_neutral=round(res["risk_neutral_default_prob"], 6),
            credit_spread_bps=round(res["implied_credit_spread_bps"], 2),
            structural_merton_spread_bps=round(res["structural_merton_spread_bps"], 2),
            solver_converged=bool(res["solver_converged"]),
            solver_used=str(res["solver_used"]),
            residual_norm=float(res["residual_norm"]),
            solver_message=f"Merton solver ({res['solver_used']}) converged (residual_norm={res['residual_norm']:.2e})",
            solver_time_ms=round(elapsed_ms, 3),
            leverage_ratio=round(res["leverage_ratio"], 4)
        )


def solve_merton(merton_in: MertonInput) -> MertonOutput:
    """Convenience functional wrapper for MertonSolver."""
    return MertonSolver.solve(merton_in)
