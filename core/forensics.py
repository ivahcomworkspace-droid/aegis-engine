"""
Aegis Quantitative Core: Financial Statement Forensics & Solvency Auditing.
Implements the 8-Factor Beneish M-Score, 4-Variable Altman Z''-Score, and CCC Working Capital Drag.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from scipy.stats import norm

try:
    from aegis_engine.schemas.credit_models import (
        ForensicInput,
        BeneishResult,
        AltmanResult,
        CCCResult,
        ForensicAnalysisResult,
    )
except ModuleNotFoundError:
    from schemas.credit_models import (
        ForensicInput,
        BeneishResult,
        AltmanResult,
        CCCResult,
        ForensicAnalysisResult,
    )


def calculate_beneish_m_score(
    sales_t: float, sales_prev: float,
    cogs_t: float, cogs_prev: float,
    rec_t: float, rec_prev: float,
    ca_t: float, ca_prev: float,
    ppe_t: float, ppe_prev: float,
    sec_t: float, sec_prev: float,
    ta_t: float, ta_prev: float,
    dep_t: float, dep_prev: float,
    sga_t: float, sga_prev: float,
    net_income_t: float, cfo_t: float,
    total_debt_t: float, total_debt_prev: float
) -> Dict[str, Any]:
    """
    Computes the 8-Factor Beneish M-Score (Beneish 1999, 8-variable model) and flags potential earnings manipulation.
    Formula:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    Includes zero-division guards across comparative accounting periods.
    """
    # Guard against zero denominators in primary statement lines
    sales_t = max(1e-4, float(sales_t))
    sales_prev = max(1e-4, float(sales_prev))
    ta_t = max(1e-4, float(ta_t))
    ta_prev = max(1e-4, float(ta_prev))

    # 1. Days Sales in Receivables Index (DSRI)
    rec_ratio_t = rec_t / sales_t
    rec_ratio_prev = rec_prev / sales_prev
    dsri = rec_ratio_t / rec_ratio_prev if rec_ratio_prev > 0 else 1.0

    # 2. Gross Margin Index (GMI)
    gm_prev = (sales_prev - cogs_prev) / sales_prev
    gm_t = (sales_t - cogs_t) / sales_t
    gmi = gm_prev / gm_t if gm_t > 0 else 1.0

    # 3. Asset Quality Index (AQI)
    aqi_num = 1.0 - ((ca_t + ppe_t + sec_t) / ta_t)
    aqi_den = 1.0 - ((ca_prev + ppe_prev + sec_prev) / ta_prev)
    aqi = aqi_num / aqi_den if abs(aqi_den) > 1e-6 else 1.0

    # 4. Sales Growth Index (SGI)
    sgi = sales_t / sales_prev

    # 5. Depreciation Index (DEPI)
    depr_base_prev = ppe_prev + dep_prev
    depr_base_t = ppe_t + dep_t
    depr_rate_prev = dep_prev / depr_base_prev if depr_base_prev > 0 else 0.0
    depr_rate_t = dep_t / depr_base_t if depr_base_t > 0 else 0.0
    depi = depr_rate_prev / depr_rate_t if depr_rate_t > 0 else 1.0

    # 6. Sales, General, and Administrative expenses Index (SGAI)
    sgai_t = sga_t / sales_t
    sgai_prev = sga_prev / sales_prev
    sgai = sgai_t / sgai_prev if sgai_prev > 0 else 1.0

    # 7. Leverage Index (LVGI)
    lev_t = total_debt_t / ta_t
    lev_prev = total_debt_prev / ta_prev
    lvgi = lev_t / lev_prev if lev_prev > 0 else 1.0

    # 8. Total Accruals to Total Assets (TATA)
    tata = (net_income_t - cfo_t) / ta_t

    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    is_manipulator = bool(m_score > -1.78)

    red_flags = {
        "DSRI_elevated": bool(dsri > 1.15),
        "GMI_deteriorating": bool(gmi > 1.10),
        "AQI_capitalizing_expenses": bool(aqi > 1.10),
        "SGI_aggressive_growth": bool(sgi > 1.30),
        "DEPI_depreciation_slowdown": bool(depi > 1.05),
        "SGAI_overhead_surge": bool(sgai > 1.10),
        "LVGI_debt_loading": bool(lvgi > 1.15),
        "TATA_accrual_disconnect": bool(tata > 0.03)
    }

    return {
        "m_score": float(m_score),
        "is_manipulator": is_manipulator,
        "manipulation_threshold": -1.78,
        "indices": {
            "DSRI": float(dsri),
            "GMI": float(gmi),
            "AQI": float(aqi),
            "SGI": float(sgi),
            "DEPI": float(depi),
            "SGAI": float(sgai),
            "LVGI": float(lvgi),
            "TATA": float(tata)
        },
        "red_flags": red_flags
    }


def calculate_beneish(inp: ForensicInput) -> BeneishResult:
    """
    Computes the standard 8-factor Beneish M-Score from Pydantic ForensicInput (Beneish 1999, 8-variable model).
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
        - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    Flag: M-Score > -1.78 indicates high likelihood of accounting manipulation.
    """
    raw_res = calculate_beneish_m_score(
        sales_t=inp.sales_t,
        sales_prev=inp.sales_t_minus_1,
        cogs_t=inp.cogs_t,
        cogs_prev=inp.cogs_t_minus_1,
        rec_t=inp.receivables_t,
        rec_prev=inp.receivables_t_minus_1,
        ca_t=inp.current_assets_t,
        ca_prev=inp.current_assets_t_minus_1,
        ppe_t=inp.ppe_t,
        ppe_prev=inp.ppe_t_minus_1,
        sec_t=inp.securities_t,
        sec_prev=inp.securities_t_minus_1,
        ta_t=inp.total_assets_t,
        ta_prev=inp.total_assets_t_minus_1,
        dep_t=inp.depreciation_t,
        dep_prev=inp.depreciation_t_minus_1,
        sga_t=inp.sga_t,
        sga_prev=inp.sga_t_minus_1,
        net_income_t=inp.net_income_t,
        cfo_t=inp.cash_flow_operations_t,
        total_debt_t=inp.total_liabilities_t,
        total_debt_prev=inp.total_liabilities_t_minus_1,
    )

    indices = raw_res["indices"]
    dsri = indices["DSRI"]
    gmi = indices["GMI"]
    aqi = indices["AQI"]
    sgi = indices["SGI"]
    depi = indices["DEPI"]
    sgai = indices["SGAI"]
    lvgi = indices["LVGI"]
    tata = indices["TATA"]
    m_score = raw_res["m_score"]
    is_manipulator = raw_res["is_manipulator"]
    manipulation_probability = float(norm.cdf(m_score))

    red_flags: List[str] = []
    if dsri > 1.15:
        red_flags.append(f"DSRI elevated ({dsri:.2f} > 1.15): Disproportionate spike in uncollected receivables vs sales")
    if gmi > 1.10:
        red_flags.append(f"GMI elevated ({gmi:.2f} > 1.10): Compressing gross margin heightens earnings management pressure")
    if aqi > 1.10:
        red_flags.append(f"AQI elevated ({aqi:.2f} > 1.10): Potential capitalization of operational expenses / intangible asset inflation")
    if sgi > 1.30:
        red_flags.append(f"SGI high ({sgi:.2f} > 1.30): Rapid top-line expansion creates aggressive accounting vulnerabilities")
    if depi > 1.05:
        red_flags.append(f"DEPI elevated ({depi:.2f} > 1.05): Slowdown in asset depreciation rate / extending useful asset lives")
    if sgai > 1.10:
        red_flags.append(f"SGAI elevated ({sgai:.2f} > 1.10): Deteriorating overhead administrative efficiency")
    if lvgi > 1.15:
        red_flags.append(f"LVGI elevated ({lvgi:.2f} > 1.15): Balance sheet re-leveraging introduces financial strain")
    if tata > 0.03:
        red_flags.append(f"TATA positive divergence ({tata:.3f} > 0.03): Significant accrual component; cash flow lags reported earnings")

    if is_manipulator:
        red_flags.insert(0, f"CRITICAL: Beneish M-Score ({m_score:.2f} > -1.78 threshold) flags high probability of financial statement manipulation.")

    return BeneishResult(
        dsri=round(float(dsri), 4),
        gmi=round(float(gmi), 4),
        aqi=round(float(aqi), 4),
        sgi=round(float(sgi), 4),
        depi=round(float(depi), 4),
        sgai=round(float(sgai), 4),
        lvgi=round(float(lvgi), 4),
        tata=round(float(tata), 4),
        m_score=round(float(m_score), 4),
        manipulation_probability=round(manipulation_probability, 4),
        is_manipulator=is_manipulator,
        red_flags=red_flags,
    )


def calculate_altman_z_double_prime(
    working_capital: Union[float, ForensicInput],
    retained_earnings: Optional[float] = None,
    ebit: Optional[float] = None,
    book_value_equity: Optional[float] = None,
    total_assets: Optional[float] = None,
    total_liabilities: Optional[float] = None
) -> Union[Dict[str, Any], AltmanResult]:
    """
    Evaluates corporate solvency via the 4-Variable Altman Z''-Score (1993 model),
    calibrated for service firms, tech companies, and emerging markets.
    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    Supports both raw float arguments returning Dict[str, Any] and Pydantic ForensicInput
    returning AltmanResult.
    """
    if hasattr(working_capital, "total_assets_t") or isinstance(working_capital, ForensicInput):
        inp: ForensicInput = working_capital
        wc = inp.working_capital_t if inp.working_capital_t is not None else (inp.current_assets_t - inp.current_liabilities_t)
        ta = inp.total_assets_t
        tl = max(inp.total_liabilities_t, 1e-9)

        x1 = wc / ta
        x2 = inp.retained_earnings_t / ta
        x3 = inp.ebit_t / ta
        x4 = inp.book_value_equity_t / tl

        z_double_prime = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

        if z_double_prime > 2.60:
            zone = "Safe"
            distress_assessment = "Low probability of solvency distress over a 2-year horizon."
        elif z_double_prime >= 1.10:
            zone = "Grey"
            distress_assessment = "Marginal solvency headroom; enhanced covenant monitoring advised."
        else:
            zone = "Distress"
            distress_assessment = "Substantial bankruptcy / restructuring probability within 24 months."

        return AltmanResult(
            x1=round(float(x1), 4),
            x2=round(float(x2), 4),
            x3=round(float(x3), 4),
            x4=round(float(x4), 4),
            z_double_prime=round(float(z_double_prime), 4),
            zone=zone,
            distress_assessment=distress_assessment,
        )

    # Raw numerical evaluation
    if total_assets is None or total_liabilities is None or total_assets <= 0.0 or total_liabilities <= 0.0:
        raise ValueError("Total Assets and Total Liabilities must be strictly positive.")

    x1 = float(working_capital) / float(total_assets)
    x2 = float(retained_earnings) / float(total_assets)
    x3 = float(ebit) / float(total_assets)
    x4 = float(book_value_equity) / float(total_liabilities)

    z_double_prime = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    if z_double_prime > 2.60:
        zone = "SAFE"
        risk_label = "Low Insolvency Risk"
    elif z_double_prime >= 1.10:
        zone = "GREY"
        risk_label = "Elevated Surveillance Required"
    else:
        zone = "DISTRESS"
        risk_label = "High Distress / Near-Term Reorganization"

    return {
        "z_double_prime": float(z_double_prime),
        "zone": zone,
        "risk_label": risk_label,
        "components": {
            "x1_working_capital_to_ta": float(x1),
            "x2_retained_earnings_to_ta": float(x2),
            "x3_ebit_to_ta": float(x3),
            "x4_equity_to_liabilities": float(x4)
        }
    }


def calculate_cash_conversion_cycle(
    receivables: float,
    inventory: float,
    payables: float,
    sales: float,
    cogs: float,
    prev_receivables: Optional[float] = None,
    prev_inventory: Optional[float] = None,
    prev_payables: Optional[float] = None,
    prev_sales: Optional[float] = None,
    prev_cogs: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates Days Sales Outstanding (DSO), Days Inventory Outstanding (DIO),
    Days Payables Outstanding (DPO), and the complete Cash Conversion Cycle (CCC).
    Detects working capital drag and accrual decoupling alerts.
    """
    sales_curr = max(1e-4, float(sales))
    cogs_curr = max(1e-4, float(cogs))

    dso = (float(receivables) / sales_curr) * 365.0
    dio = (float(inventory) / cogs_curr) * 365.0
    dpo = (float(payables) / cogs_curr) * 365.0
    ccc = dso + dio - dpo

    delta_ccc = 0.0
    accrual_alert = False

    if all(v is not None for v in [prev_receivables, prev_inventory, prev_payables, prev_sales, prev_cogs]):
        sales_p = max(1e-4, float(prev_sales))
        cogs_p = max(1e-4, float(prev_cogs))

        dso_prev = (float(prev_receivables) / sales_p) * 365.0
        dio_prev = (float(prev_inventory) / cogs_p) * 365.0
        dpo_prev = (float(prev_payables) / cogs_p) * 365.0
        ccc_prev = dso_prev + dio_prev - dpo_prev

        delta_ccc = ccc - ccc_prev
        gm_curr = (sales_curr - cogs_curr) / sales_curr
        gm_prev = (sales_p - cogs_p) / sales_p

        # Trigger if CCC expands > 20 days while gross margin is static or expanding (|Delta GM| <= 0.02)
        if delta_ccc > 20.0 and abs(gm_curr - gm_prev) <= 0.02:
            accrual_alert = True

    return {
        "dso_days": float(dso),
        "dio_days": float(dio),
        "dpo_days": float(dpo),
        "ccc_days": float(ccc),
        "delta_ccc_days": float(delta_ccc),
        "accrual_divergence_alert": accrual_alert
    }


def calculate_ccc(inp: ForensicInput) -> CCCResult:
    """
    Computes Cash Conversion Cycle (CCC = DSO + DIO - DPO) across period t and t-1.
    Triggers an alert if Delta CCC > +20 days while Gross Margin is static (|Delta GM| <= 0.02).
    """
    raw_res = calculate_cash_conversion_cycle(
        receivables=inp.receivables_t,
        inventory=inp.inventory_t,
        payables=inp.accounts_payable_t,
        sales=inp.sales_t,
        cogs=inp.cogs_t,
        prev_receivables=inp.receivables_t_minus_1,
        prev_inventory=inp.inventory_t_minus_1,
        prev_payables=inp.accounts_payable_t_minus_1,
        prev_sales=inp.sales_t_minus_1,
        prev_cogs=inp.cogs_t_minus_1,
    )

    dso_t = raw_res["dso_days"]
    dio_t = raw_res["dio_days"]
    dpo_t = raw_res["dpo_days"]
    ccc_t = raw_res["ccc_days"]
    delta_ccc = raw_res["delta_ccc_days"]
    warning_triggered = raw_res["accrual_divergence_alert"]

    # Historical metrics for detailed report
    sales_p = max(1e-4, float(inp.sales_t_minus_1))
    cogs_p = max(1e-4, float(inp.cogs_t_minus_1))
    dso_t_minus_1 = (float(inp.receivables_t_minus_1) / sales_p) * 365.0
    dio_t_minus_1 = (float(inp.inventory_t_minus_1) / cogs_p) * 365.0
    dpo_t_minus_1 = (float(inp.accounts_payable_t_minus_1) / cogs_p) * 365.0
    ccc_t_minus_1 = dso_t_minus_1 + dio_t_minus_1 - dpo_t_minus_1

    gm_t = (inp.sales_t - inp.cogs_t) / inp.sales_t
    gm_t_minus_1 = (inp.sales_t_minus_1 - inp.cogs_t_minus_1) / inp.sales_t_minus_1
    delta_gm = gm_t - gm_t_minus_1

    alert_message = None
    if warning_triggered:
        alert_message = (
            f"WORKING CAPITAL STRESS ALERT: Cash Conversion Cycle stretched by {delta_ccc:+.1f} days "
            f"(from {ccc_t_minus_1:.1f} to {ccc_t:.1f} days) while gross margin is static "
            f"({delta_gm * 100:+.2f}%). Signals trapped liquidity and unbilled revenue accumulation."
        )

    return CCCResult(
        dso_t=round(float(dso_t), 2),
        dio_t=round(float(dio_t), 2),
        dpo_t=round(float(dpo_t), 2),
        ccc_t=round(float(ccc_t), 2),
        dso_t_minus_1=round(float(dso_t_minus_1), 2),
        dio_t_minus_1=round(float(dio_t_minus_1), 2),
        dpo_t_minus_1=round(float(dpo_t_minus_1), 2),
        ccc_t_minus_1=round(float(ccc_t_minus_1), 2),
        delta_ccc=round(float(delta_ccc), 2),
        delta_gross_margin=round(float(delta_gm), 4),
        warning_triggered=warning_triggered,
        alert_message=alert_message,
    )


def calculate_forensics(inp: ForensicInput) -> ForensicAnalysisResult:
    """
    Executes unified balance sheet forensic audit.
    Synthesizes Beneish M-Score, Altman Z''-Score, and CCC dynamics into a composite risk tier.
    """
    beneish = calculate_beneish(inp)
    altman = calculate_altman_z_double_prime(inp)
    ccc = calculate_ccc(inp)

    # Determine composite forensic risk
    risk_score = 0
    findings: List[str] = []

    if beneish.is_manipulator:
        risk_score += 3
        findings.append(f"Active earnings manipulation warning: Beneish M-Score = {beneish.m_score:.2f} (> -1.78).")
    elif beneish.m_score > -2.22:
        risk_score += 1
        findings.append(f"Beneish M-Score in borderline elevated zone ({beneish.m_score:.2f}).")

    if altman.zone in ["Distress", "DISTRESS"]:
        risk_score += 3
        findings.append(f"Altman Z'' indicates severe distress ({altman.z_double_prime:.2f} < 1.10).")
    elif altman.zone in ["Grey", "GREY"]:
        risk_score += 1
        findings.append(f"Altman Z'' in grey indeterminacy zone ({altman.z_double_prime:.2f}).")
    else:
        findings.append(f"Altman Z'' in safe zone ({altman.z_double_prime:.2f} > 2.60).")

    if ccc.warning_triggered:
        risk_score += 2
        findings.append(ccc.alert_message or "CCC stretched significantly without gross margin justification.")
    elif ccc.delta_ccc > 10.0:
        findings.append(f"Working capital cycle elongating: CCC increased by {ccc.delta_ccc:+.1f} days.")

    # Assign tier
    if risk_score >= 4:
        composite_tier = "Critical"
    elif risk_score >= 2:
        composite_tier = "High"
    elif risk_score >= 1:
        composite_tier = "Medium"
    else:
        composite_tier = "Low"

    return ForensicAnalysisResult(
        beneish=beneish,
        altman=altman,
        ccc=ccc,
        composite_forensic_risk=composite_tier,
        summary_findings=findings,
    )
