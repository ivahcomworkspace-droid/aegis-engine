"""
Aegis Regulatory Loss Distribution & Stress Testing Engine
Implements:
1. Vasicek Asymptotic Single Risk Factor (ASRF) conditional default distribution and Basel IRB capital.
2. Theoretical Vasicek continuous loss probability density distribution.
3. Multi-period lifetime Expected Credit Loss (ECL) under IFRS 9 / CECL with 3-stage transitions.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from scipy.stats import norm

try:
    from aegis_engine.schemas.credit_models import (
        VasicekInput,
        VasicekOutput,
        IFRS9StageInput,
        ECLPeriodSchedule,
        ECLResult,
    )
except ModuleNotFoundError:
    from schemas.credit_models import (
        VasicekInput,
        VasicekOutput,
        IFRS9StageInput,
        ECLPeriodSchedule,
        ECLResult,
    )


def calculate_corporate_asset_correlation(
    pd: float,
    firm_sales_eur_m: Optional[float] = None
) -> float:
    """
    Computes Basel corporate asset correlation rho:
    rho = 0.12 * (1 - exp(-50 * PD)) / (1 - exp(-50)) + 0.24 * [1 - (1 - exp(-50 * PD)) / (1 - exp(-50))]
    If firm sales are specified between 5M and 50M EUR, applies Basel SME size adjustment:
    Delta rho = -0.04 * (1 - (Sales - 5) / 45)
    Bound to [0.12, 0.24].
    """
    pd_clamped = min(max(float(pd), 1e-6), 0.999999)
    exp_factor = (1.0 - np.exp(-50.0 * pd_clamped)) / (1.0 - np.exp(-50.0))
    rho = 0.12 * exp_factor + 0.24 * (1.0 - exp_factor)

    if firm_sales_eur_m is not None:
        sales_clamped = min(max(float(firm_sales_eur_m), 5.0), 50.0)
        sme_adjustment = -0.04 * (1.0 - (sales_clamped - 5.0) / 45.0)
        rho += sme_adjustment

    return float(np.clip(rho, 0.12, 0.24))


def calculate_vasicek_conditional_pd(pd: float, rho: float, z: float) -> float:
    """
    Vasicek conditional default probability:
    PD_conditional(Z) = Phi( [Phi^-1(PD) + sqrt(rho) * Z] / sqrt(1 - rho) )
    Where Z is standard normal systemic shock (Z > 0 denotes economic downturn).
    Guarded against boundary singularities on PD and rho.
    """
    pd_clamped = min(max(float(pd), 1e-7), 1.0 - 1e-7)
    rho_clamped = min(max(float(rho), 1e-5), 1.0 - 1e-5)

    inv_norm_pd = norm.ppf(pd_clamped)
    sqrt_rho = np.sqrt(rho_clamped)
    sqrt_one_minus_rho = np.sqrt(1.0 - rho_clamped)

    arg = (inv_norm_pd + sqrt_rho * float(z)) / sqrt_one_minus_rho
    return float(norm.cdf(arg))


def calculate_vasicek_loss_distribution(
    pd: float,
    rho: float,
    num_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes loss rate array and theoretical Vasicek probability density function f(x):
    f(x) = sqrt((1-rho)/rho) * exp( -0.5/rho * (sqrt(1-rho)*Phi^-1(x) - Phi^-1(PD))^2 + 0.5 * (Phi^-1(x))^2 )
    Guarded against singularities at x -> 0 and x -> 1.
    """
    rho_clamped = min(max(float(rho), 1e-4), 0.999)
    pd_clamped = min(max(float(pd), 1e-6), 1.0 - 1e-6)

    # Sample loss rates away from 0 and 1 singularities
    x = np.linspace(0.001, 0.999, num_points)
    inv_x = norm.ppf(x)
    inv_pd = norm.ppf(pd_clamped)

    sqrt_factor = np.sqrt((1.0 - rho_clamped) / rho_clamped)
    exponent = - (1.0 / (2.0 * rho_clamped)) * (np.sqrt(1.0 - rho_clamped) * inv_x - inv_pd)**2 + 0.5 * (inv_x**2)
    density = sqrt_factor * np.exp(exponent)

    return x, density


def calculate_vasicek_capital(
    pit_pd: float,
    lgd: float,
    ead: float,
    maturity: float = 1.0,
    confidence_level: float = 0.999,
    macro_shock_z: Optional[float] = None,
    firm_sales_eur_m: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates Basel IRB / Vasicek ASRF regulatory capital, Unexpected Loss, Expected Loss,
    and stress scenario evaluations.
    """
    pd = min(max(float(pit_pd), 1e-6), 0.999)
    lgd_val = min(max(float(lgd), 0.0), 1.0)
    ead_val = max(0.0, float(ead))
    m = max(1.0, float(maturity))

    rho = calculate_corporate_asset_correlation(pd, firm_sales_eur_m)

    z_reg = float(norm.ppf(min(max(float(confidence_level), 0.50), 0.9999)))
    z_eval = float(macro_shock_z) if macro_shock_z is not None else z_reg

    cond_pd_eval = calculate_vasicek_conditional_pd(pd, rho, z_eval)
    cond_pd_999 = calculate_vasicek_conditional_pd(pd, rho, z_reg)

    # Basel IRB maturity adjustment: b = (0.11852 - 0.05478 * ln(PD))^2
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    maturity_adj = (1.0 + (m - 2.5) * b) / max(1.0 - 1.5 * b, 0.1)

    # Capital requirement ratio K
    k_ratio = max((lgd_val * cond_pd_999 - pd * lgd_val) * maturity_adj, 0.0)

    ul_currency = k_ratio * ead_val
    el_currency = pd * lgd_val * ead_val
    total_capital = ul_currency + el_currency

    scenarios = {
        "Base (Z=0.0)": round(calculate_vasicek_conditional_pd(pd, rho, 0.0), 6),
        "Mild Downturn (Z=+1.0)": round(calculate_vasicek_conditional_pd(pd, rho, 1.0), 6),
        "Severe Downturn (Z=+2.33, 99%)": round(calculate_vasicek_conditional_pd(pd, rho, 2.3263), 6),
        f"Regulatory VaR (Z=+{z_reg:.2f}, {confidence_level*100:.1f}%)": round(cond_pd_999, 6),
    }

    return {
        "pit_pd": round(pd, 6),
        "asset_correlation_rho": round(rho, 6),
        "conditional_pd": round(cond_pd_eval, 6),
        "regulatory_capital_k": round(float(k_ratio), 6),
        "unexpected_loss": round(float(ul_currency), 2),
        "expected_loss": round(float(el_currency), 2),
        "economic_capital_ead": round(float(total_capital), 2),
        "stress_scenarios": scenarios,
    }


def calculate_vasicek_asrf(inp: VasicekInput) -> VasicekOutput:
    """
    Computes full Basel IRB / Vasicek ASRF regulatory capital and loss metrics from Pydantic input.
    """
    res = calculate_vasicek_capital(
        pit_pd=inp.pit_pd,
        lgd=inp.lgd,
        ead=inp.ead,
        maturity=inp.maturity,
        confidence_level=inp.confidence_level,
        macro_shock_z=inp.macro_shock_z,
        firm_sales_eur_m=inp.firm_sales_eur_millions,
    )

    return VasicekOutput(
        pit_pd=res["pit_pd"],
        asset_correlation_rho=res["asset_correlation_rho"],
        conditional_pd=res["conditional_pd"],
        regulatory_capital_k=res["regulatory_capital_k"],
        unexpected_loss=res["unexpected_loss"],
        expected_loss=res["expected_loss"],
        economic_capital_ead=res["economic_capital_ead"],
        stress_scenarios=res["stress_scenarios"],
    )


def calculate_ecl_staging_and_loss(
    current_pd: float,
    origination_distance_to_default: float = 3.0,
    current_distance_to_default: float = 3.0,
    days_past_due: int = 0,
    is_credit_impaired: bool = False,
    ead_schedule: Optional[List[float]] = None,
    lgd_schedule: Optional[List[float]] = None,
    effective_interest_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates IFRS 9 / CECL staging criteria and multi-period discounted Expected Credit Loss.
    """
    delta_dd = current_distance_to_default - origination_distance_to_default
    dpd = int(days_past_due)
    pd_curr = min(max(float(current_pd), 1e-6), 0.999)
    eir = max(0.0, float(effective_interest_rate))

    ead_sched = ead_schedule if ead_schedule and len(ead_schedule) > 0 else [1_000_000.0]
    num_periods = len(ead_sched)

    # Staging Classification
    if is_credit_impaired or dpd >= 90 or pd_curr >= 0.50:
        assigned_stage = 3
        stage_name = "Stage 3 (Credit Impaired)"
        stage_rationale = (
            f"Credit Impaired trigger: DPD={dpd} (threshold >= 90) or PD={pd_curr:.1%} >= 50% "
            f"or active restructuring status. Lifetime ECL mandatory."
        )
        is_lifetime = True
        eval_periods = num_periods
    elif delta_dd < -1.0 or dpd >= 30:
        assigned_stage = 2
        stage_name = "Stage 2 (Underperforming - SICR)"
        trigger_reason = f"Delta DD = {delta_dd:+.2f}sigma (< -1.0sigma SICR threshold)" if delta_dd < -1.0 else f"DPD={dpd} in [30, 89]"
        stage_rationale = (
            f"Significant Increase in Credit Risk (SICR) triggered by {trigger_reason}. "
            f"Mandatory transition from 12-month to Lifetime ECL."
        )
        is_lifetime = True
        eval_periods = num_periods
    else:
        assigned_stage = 1
        stage_name = "Stage 1 (Performing - 12-Month ECL)"
        stage_rationale = (
            f"Stable credit profile: Delta DD = {delta_dd:+.2f}sigma (>= -1.0sigma), DPD={dpd} (< 30). "
            f"Provisioning limited to 12-Month ECL horizon."
        )
        is_lifetime = False
        eval_periods = 1

    # LGD schedule
    if lgd_schedule and len(lgd_schedule) >= num_periods:
        lgd_sched = [min(max(float(v), 0.0), 1.0) for v in lgd_schedule]
    else:
        base_lgd = 0.65 if assigned_stage == 3 else 0.45
        lgd_sched = [base_lgd] * num_periods

    if assigned_stage == 3:
        # Fix V1: Under IFRS 9 / CECL, when an asset is Stage 3 (Credit Impaired), default has already occurred (PD = 1.0).
        # Loss allowance equals the full discounted LGD * EAD for the current default event.
        ead_0 = float(ead_sched[0])
        lgd_0 = float(lgd_sched[0])
        discount_factor = 1.0 / (1.0 + eir)
        discounted_ecl_0 = lgd_0 * ead_0 * discount_factor
        total_ecl = discounted_ecl_0

        schedules = [{
            "period": 1,
            "marginal_pd": 1.0,
            "cumulative_pd": 1.0,
            "survival_probability": 0.0,
            "ead": round(ead_0, 2),
            "lgd": round(lgd_0, 4),
            "discount_factor": round(discount_factor, 6),
            "discounted_ecl": round(discounted_ecl_0, 2),
        }]
        for t in range(2, eval_periods + 1):
            idx = t - 1
            ead_t = float(ead_sched[idx])
            lgd_t = float(lgd_sched[idx])
            df_t = 1.0 / ((1.0 + eir) ** t)
            schedules.append({
                "period": t,
                "marginal_pd": 0.0,
                "cumulative_pd": 1.0,
                "survival_probability": 0.0,
                "ead": round(ead_t, 2),
                "lgd": round(lgd_t, 4),
                "discount_factor": round(df_t, 6),
                "discounted_ecl": 0.0,
            })
    else:
        # Hazard rate formulation: S(t) = exp(-lambda * t)
        lambda_hazard = -np.log(max(1.0 - pd_curr, 1e-7))

        schedules = []
        total_ecl = 0.0
        cumulative_survival = 1.0

        for t in range(1, eval_periods + 1):
            idx = t - 1
            ead_t = float(ead_sched[idx])
            lgd_t = float(lgd_sched[idx])

            survival_t = float(np.exp(-lambda_hazard * t))
            marginal_pd = cumulative_survival - survival_t
            cumulative_pd = 1.0 - survival_t
            cumulative_survival = survival_t

            discount_factor = 1.0 / ((1.0 + eir) ** t)
            discounted_ecl_t = marginal_pd * lgd_t * ead_t * discount_factor
            total_ecl += discounted_ecl_t

            schedules.append({
                "period": t,
                "marginal_pd": round(float(marginal_pd), 6),
                "cumulative_pd": round(float(cumulative_pd), 6),
                "survival_probability": round(float(survival_t), 6),
                "ead": round(float(ead_t), 2),
                "lgd": round(float(lgd_t), 4),
                "discount_factor": round(float(discount_factor), 6),
                "discounted_ecl": round(float(discounted_ecl_t), 2),
            })

    initial_ead = float(ead_sched[0])
    ecl_pct = (total_ecl / max(initial_ead, 1e-9)) * 100.0

    return {
        "assigned_stage": assigned_stage,
        "stage_name": stage_name,
        "stage_rationale": stage_rationale,
        "delta_distance_to_default": round(float(delta_dd), 4),
        "total_ecl": round(float(total_ecl), 2),
        "is_lifetime_ecl": is_lifetime,
        "ecl_percentage_of_ead": round(float(ecl_pct), 3),
        "period_schedules": schedules,
    }


def calculate_ifrs9_ecl(stage_in: IFRS9StageInput) -> ECLResult:
    """
    Computes multi-period Expected Credit Loss (ECL) and performs IFRS 9 3-stage classification
    from Pydantic input.
    """
    raw_res = calculate_ecl_staging_and_loss(
        current_pd=stage_in.current_pd,
        origination_distance_to_default=stage_in.origination_distance_to_default,
        current_distance_to_default=stage_in.current_distance_to_default,
        days_past_due=stage_in.days_past_due,
        is_credit_impaired=stage_in.is_credit_impaired,
        ead_schedule=stage_in.ead_schedule,
        lgd_schedule=stage_in.lgd_schedule,
        effective_interest_rate=stage_in.effective_interest_rate,
    )

    schedules = [
        ECLPeriodSchedule(
            period=s["period"],
            marginal_pd=s["marginal_pd"],
            cumulative_pd=s["cumulative_pd"],
            survival_probability=s["survival_probability"],
            ead=s["ead"],
            lgd=s["lgd"],
            discount_factor=s["discount_factor"],
            discounted_ecl=s["discounted_ecl"],
        )
        for s in raw_res["period_schedules"]
    ]

    return ECLResult(
        assigned_stage=raw_res["assigned_stage"],
        stage_name=raw_res["stage_name"],
        stage_rationale=raw_res["stage_rationale"],
        delta_distance_to_default=raw_res["delta_distance_to_default"],
        total_ecl=raw_res["total_ecl"],
        is_lifetime_ecl=raw_res["is_lifetime_ecl"],
        ecl_percentage_of_ead=raw_res["ecl_percentage_of_ead"],
        period_schedules=schedules,
    )


# Functional alias for multi-period ECL computation
calculate_multi_period_ecl = calculate_ecl_staging_and_loss
