"""
Aegis Regulatory Loss Distribution & Stress Testing Engine
Implements:
1. Vasicek Asymptotic Single Risk Factor (ASRF) conditional default distribution and Basel IRB capital.
2. Macroeconomic stress scenario generation and density distribution.
3. Multi-period lifetime Expected Credit Loss (ECL) under IFRS 9 / CECL with 3-stage transitions.
"""

from typing import List, Dict, Tuple, Optional
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


def calculate_corporate_asset_correlation(pd: float, firm_sales_eur_m: Optional[float] = None) -> float:
    """
    Computes Basel corporate asset correlation rho:
    rho = 0.12 * (1 - exp(-50 * PD)) / (1 - exp(-50)) + 0.24 * [1 - (1 - exp(-50 * PD)) / (1 - exp(-50))]
    If firm sales are specified between 5M and 50M EUR, applies Basel SME size adjustment.
    """
    pd_clamped = min(max(pd, 1e-6), 0.999999)
    factor = (1.0 - np.exp(-50.0 * pd_clamped)) / (1.0 - np.exp(-50.0))
    rho = 0.12 * factor + 0.24 * (1.0 - factor)

    if firm_sales_eur_m is not None:
        sales_clamped = min(max(firm_sales_eur_m, 5.0), 50.0)
        sme_adjustment = -0.04 * (1.0 - (sales_clamped - 5.0) / 45.0)
        rho += sme_adjustment

    return float(np.clip(rho, 0.12, 0.24))


def calculate_vasicek_conditional_pd(pd: float, rho: float, z: float) -> float:
    """
    Vasicek conditional default probability:
    PD_conditional(Z) = Phi( [Phi^-1(PD) + sqrt(rho) * Z] / sqrt(1 - rho) )
    Where Z is standard normal systemic shock (Z > 0 denotes economic downturn).
    """
    pd_clamped = min(max(pd, 1e-7), 1.0 - 1e-7)
    inv_norm_pd = norm.ppf(pd_clamped)
    sqrt_rho = np.sqrt(rho)
    sqrt_one_minus_rho = np.sqrt(1.0 - rho)

    arg = (inv_norm_pd + sqrt_rho * z) / sqrt_one_minus_rho
    return float(norm.cdf(arg))


def calculate_vasicek_loss_distribution(
    pd: float,
    rho: float,
    num_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes loss rate array and theoretical Vasicek probability density function f(x):
    f(x) = sqrt((1-rho)/rho) * exp( -0.5/rho * (sqrt(1-rho)*Phi^-1(x) - Phi^-1(PD))^2 + 0.5 * (Phi^-1(x))^2 )
    """
    # Sample loss rates away from 0 and 1 singularities
    x = np.linspace(0.001, 0.999, num_points)
    inv_x = norm.ppf(x)
    inv_pd = norm.ppf(min(max(pd, 1e-6), 1.0 - 1e-6))

    sqrt_factor = np.sqrt((1.0 - rho) / rho)
    exponent = - (1.0 / (2.0 * rho)) * (np.sqrt(1.0 - rho) * inv_x - inv_pd)**2 + 0.5 * (inv_x**2)
    density = sqrt_factor * np.exp(exponent)

    return x, density


def calculate_vasicek_asrf(inp: VasicekInput) -> VasicekOutput:
    """
    Computes full Basel IRB / Vasicek ASRF regulatory capital and loss metrics.
    """
    pd = inp.pit_pd
    lgd = inp.lgd
    ead = inp.ead
    m = inp.maturity

    rho = calculate_corporate_asset_correlation(pd, inp.firm_sales_eur_millions)

    # Standard stress shocks Z:
    # Basel 99.9% VaR shock is Phi^-1(0.999) ~ 3.0902
    z_reg = float(norm.ppf(inp.confidence_level))
    z_eval = inp.macro_shock_z if inp.macro_shock_z is not None else z_reg

    cond_pd_eval = calculate_vasicek_conditional_pd(pd, rho, z_eval)

    # Basel IRB maturity adjustment: b = (0.11852 - 0.05478 * ln(PD))^2
    b = (0.11852 - 0.05478 * np.log(min(max(pd, 1e-5), 0.99))) ** 2
    maturity_adj = (1.0 + (m - 2.5) * b) / max(1.0 - 1.5 * b, 0.1)

    # Regulatory Capital Requirement ratio K
    # K = [LGD * PD_cond(99.9%) - PD * LGD] * MaturityAdjustment
    cond_pd_999 = calculate_vasicek_conditional_pd(pd, rho, z_reg)
    k_ratio = max((lgd * cond_pd_999 - pd * lgd) * maturity_adj, 0.0)

    # Unexpected Loss & Expected Loss
    ul_currency = k_ratio * ead
    el_currency = pd * lgd * ead
    total_capital = ul_currency + el_currency

    # Scenario grid
    scenarios = {
        "Base (Z=0.0)": round(calculate_vasicek_conditional_pd(pd, rho, 0.0), 6),
        "Mild Downturn (Z=+1.0)": round(calculate_vasicek_conditional_pd(pd, rho, 1.0), 6),
        "Severe Downturn (Z=+2.33, 99%)": round(calculate_vasicek_conditional_pd(pd, rho, 2.3263), 6),
        f"Regulatory VaR (Z=+{z_reg:.2f}, {inp.confidence_level*100:.1f}%)": round(cond_pd_999, 6),
    }

    return VasicekOutput(
        pit_pd=round(pd, 6),
        asset_correlation_rho=round(rho, 6),
        conditional_pd=round(cond_pd_eval, 6),
        regulatory_capital_k=round(float(k_ratio), 6),
        unexpected_loss=round(float(ul_currency), 2),
        expected_loss=round(float(el_currency), 2),
        economic_capital_ead=round(float(total_capital), 2),
        stress_scenarios=scenarios,
    )


def calculate_ifrs9_ecl(stage_in: IFRS9StageInput) -> ECLResult:
    """
    Computes multi-period Expected Credit Loss (ECL) and performs IFRS 9 3-stage classification.
    ECL = sum_{t=1}^T [ PD_t^* * LGD_t * EAD_t / (1 + EIR)^t ]

    IFRS 9 Staging Criteria:
    - Stage 3: Credit Impaired (is_credit_impaired is True, DPD >= 90, or PD >= 0.50)
    - Stage 2: Significant Increase in Credit Risk (SICR):
               Delta DD < -1.0 sigma (DD_current - DD_origination < -1.0)
               OR DPD in [30, 89]
    - Stage 1: Performing (12-Month ECL, T = 1)
    """
    delta_dd = stage_in.current_distance_to_default - stage_in.origination_distance_to_default
    current_pd = stage_in.current_pd
    dpd = stage_in.days_past_due
    eir = stage_in.effective_interest_rate
    ead_sched = stage_in.ead_schedule
    num_periods = len(ead_sched)

    # Determine default staging
    if stage_in.is_credit_impaired or dpd >= 90 or current_pd >= 0.50:
        assigned_stage = 3
        stage_name = "Stage 3 (Credit Impaired)"
        stage_rationale = (
            f"Credit Impaired trigger: DPD={dpd} (threshold >= 90) or PD={current_pd:.1%} >= 50% "
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
        eval_periods = 1  # 12-month horizon only

    # Build LGD schedule
    if stage_in.lgd_schedule and len(stage_in.lgd_schedule) >= num_periods:
        lgd_sched = stage_in.lgd_schedule
    else:
        base_lgd = 0.65 if assigned_stage == 3 else 0.45
        lgd_sched = [base_lgd] * num_periods

    # Hazard rate formulation for term structure
    # lambda_h = -ln(1 - PD_1)
    lambda_hazard = -np.log(max(1.0 - min(current_pd, 0.999), 1e-7))

    schedules: List[ECLPeriodSchedule] = []
    total_ecl = 0.0

    cumulative_survival = 1.0

    for t in range(1, eval_periods + 1):
        idx = t - 1
        ead_t = ead_sched[idx]
        lgd_t = lgd_sched[idx]

        # Survival at end of period t: S(t) = exp(-lambda * t)
        survival_t = np.exp(-lambda_hazard * t)
        marginal_pd = cumulative_survival - survival_t
        cumulative_pd = 1.0 - survival_t
        cumulative_survival = survival_t

        # Discount factor (1 + EIR)^-t
        discount_factor = 1.0 / ((1.0 + eir) ** t)
        discounted_ecl_t = marginal_pd * lgd_t * ead_t * discount_factor
        total_ecl += discounted_ecl_t

        schedules.append(
            ECLPeriodSchedule(
                period=t,
                marginal_pd=round(float(marginal_pd), 6),
                cumulative_pd=round(float(cumulative_pd), 6),
                survival_probability=round(float(survival_t), 6),
                ead=round(float(ead_t), 2),
                lgd=round(float(lgd_t), 4),
                discount_factor=round(float(discount_factor), 6),
                discounted_ecl=round(float(discounted_ecl_t), 2),
            )
        )

    initial_ead = ead_sched[0]
    ecl_pct = (total_ecl / max(initial_ead, 1e-9)) * 100.0

    return ECLResult(
        assigned_stage=assigned_stage,
        stage_name=stage_name,
        stage_rationale=stage_rationale,
        delta_distance_to_default=round(float(delta_dd), 4),
        total_ecl=round(float(total_ecl), 2),
        is_lifetime_ecl=is_lifetime,
        ecl_percentage_of_ead=round(float(ecl_pct), 3),
        period_schedules=schedules,
    )
