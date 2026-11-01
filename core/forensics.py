"""
Aegis Balance Sheet Forensics Engine
Implements:
1. 8-Factor Beneish M-Score for earnings manipulation detection.
2. Altman Z''-Score for services, non-manufacturers, and emerging market distress analysis.
3. Cash Conversion Cycle (CCC) dynamics and working capital stress alerts.
"""

from typing import List, Tuple
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


def calculate_beneish(inp: ForensicInput) -> BeneishResult:
    """
    Computes the standard 8-factor Beneish M-Score.
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
        - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Flag: M-Score > -1.78 indicates high likelihood of accounting manipulation.
    """
    red_flags: List[str] = []

    # 1. DSRI: Days Sales in Receivables Index
    # (Receivables_t / Sales_t) / (Receivables_{t-1} / Sales_{t-1})
    rec_to_sales_t = inp.receivables_t / inp.sales_t
    rec_to_sales_t_minus_1 = inp.receivables_t_minus_1 / inp.sales_t_minus_1
    dsri = rec_to_sales_t / max(rec_to_sales_t_minus_1, 1e-9)
    if dsri > 1.25:
        red_flags.append(f"DSRI elevated ({dsri:.2f} > 1.25): Disproportionate spike in uncollected receivables vs sales")

    # 2. GMI: Gross Margin Index
    # [(Sales_{t-1} - COGS_{t-1}) / Sales_{t-1}] / [(Sales_t - COGS_t) / Sales_t]
    margin_t_minus_1 = (inp.sales_t_minus_1 - inp.cogs_t_minus_1) / inp.sales_t_minus_1
    margin_t = (inp.sales_t - inp.cogs_t) / inp.sales_t
    gmi = margin_t_minus_1 / max(margin_t, 1e-9)
    if gmi > 1.10:
        red_flags.append(f"GMI elevated ({gmi:.2f} > 1.10): Compressing gross margin heightens earnings management pressure")

    # 3. AQI: Asset Quality Index
    # [1 - (CurrentAssets_t + PPE_t + Securities_t) / TotalAssets_t] / [1 - (CurrentAssets_{t-1} + PPE_{t-1} + Securities_{t-1}) / TotalAssets_{t-1}]
    non_hard_assets_t = 1.0 - (inp.current_assets_t + inp.ppe_t + inp.securities_t) / inp.total_assets_t
    non_hard_assets_t_minus_1 = 1.0 - (inp.current_assets_t_minus_1 + inp.ppe_t_minus_1 + inp.securities_t_minus_1) / inp.total_assets_t_minus_1
    aqi = max(non_hard_assets_t, 0.0) / max(non_hard_assets_t_minus_1, 1e-9)
    if aqi > 1.25:
        red_flags.append(f"AQI elevated ({aqi:.2f} > 1.25): Potential capitalization of operational expenses / intangible asset inflation")

    # 4. SGI: Sales Growth Index
    # Sales_t / Sales_{t-1}
    sgi = inp.sales_t / inp.sales_t_minus_1
    if sgi > 1.30:
        red_flags.append(f"SGI high ({sgi:.2f} > 1.30): Rapid top-line expansion creates aggressive accounting vulnerabilities")

    # 5. DEPI: Depreciation Index
    # DeprRate_{t-1} / DeprRate_t where DeprRate = Depr / (PPE + Depr)
    depr_rate_t_minus_1 = inp.depreciation_t_minus_1 / max(inp.ppe_t_minus_1 + inp.depreciation_t_minus_1, 1e-9)
    depr_rate_t = inp.depreciation_t / max(inp.ppe_t + inp.depreciation_t, 1e-9)
    depi = depr_rate_t_minus_1 / max(depr_rate_t, 1e-9)
    if depi > 1.10:
        red_flags.append(f"DEPI elevated ({depi:.2f} > 1.10): Slowdown in asset depreciation rate / extending useful asset lives")

    # 6. SGAI: Sales, General & Administrative Expenses Index
    # (SGA_t / Sales_t) / (SGA_{t-1} / Sales_{t-1})
    sga_ratio_t = inp.sga_t / inp.sales_t
    sga_ratio_t_minus_1 = inp.sga_t_minus_1 / inp.sales_t_minus_1
    sgai = sga_ratio_t / max(sga_ratio_t_minus_1, 1e-9)
    if sgai > 1.15:
        red_flags.append(f"SGAI elevated ({sgai:.2f} > 1.15): Deteriorating overhead administrative efficiency")

    # 7. LVGI: Leverage Index
    # [TotalDebt_t / TotalAssets_t] / [TotalDebt_{t-1} / TotalAssets_{t-1}]
    # Using total liabilities or total debt
    lev_t = inp.total_liabilities_t / inp.total_assets_t
    lev_t_minus_1 = inp.total_liabilities_t_minus_1 / inp.total_assets_t_minus_1
    lvgi = lev_t / max(lev_t_minus_1, 1e-9)
    if lvgi > 1.20:
        red_flags.append(f"LVGI elevated ({lvgi:.2f} > 1.20): Balance sheet re-leveraging introduces financial strain")

    # 8. TATA: Total Accruals to Total Assets
    # (NetIncome_t - CFO_t) / TotalAssets_t
    tata = (inp.net_income_t - inp.cash_flow_operations_t) / inp.total_assets_t
    if tata > 0.05:
        red_flags.append(f"TATA positive divergence ({tata:.3f} > 0.05): Significant accrual component; cash flow lags reported earnings")

    # 8-Factor Beneish equation
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )

    is_manipulator = bool(m_score > -1.78)
    manipulation_probability = float(norm.cdf(m_score))

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


def calculate_altman_z_double_prime(inp: ForensicInput) -> AltmanResult:
    """
    Computes Altman Z''-Score for services, non-manufacturers, and emerging markets.
    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Book Value of Equity / Total Liabilities
    Zones:
        Safe: Z'' > 2.60
        Grey: 1.10 <= Z'' <= 2.60
        Distress: Z'' < 1.10
    """
    wc = inp.working_capital_t if inp.working_capital_t is not None else (inp.current_assets_t - inp.current_liabilities_t)
    total_assets = inp.total_assets_t
    total_liabilities = max(inp.total_liabilities_t, 1e-9)

    x1 = wc / total_assets
    x2 = inp.retained_earnings_t / total_assets
    x3 = inp.ebit_t / total_assets
    x4 = inp.book_value_equity_t / total_liabilities

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


def calculate_ccc(inp: ForensicInput) -> CCCResult:
    """
    Computes Cash Conversion Cycle (CCC = DSO + DIO - DPO) across period t and t-1.
    Triggers an alert if Delta CCC > +20 days while Gross Margin is static (|Delta GM| <= 0.02).
    """
    # Period t
    dso_t = (inp.receivables_t / inp.sales_t) * 365.0
    dio_t = (inp.inventory_t / max(inp.cogs_t, 1e-9)) * 365.0
    dpo_t = (inp.accounts_payable_t / max(inp.cogs_t, 1e-9)) * 365.0
    ccc_t = dso_t + dio_t - dpo_t

    # Period t-1
    dso_t_minus_1 = (inp.receivables_t_minus_1 / inp.sales_t_minus_1) * 365.0
    dio_t_minus_1 = (inp.inventory_t_minus_1 / max(inp.cogs_t_minus_1, 1e-9)) * 365.0
    dpo_t_minus_1 = (inp.accounts_payable_t_minus_1 / max(inp.cogs_t_minus_1, 1e-9)) * 365.0
    ccc_t_minus_1 = dso_t_minus_1 + dio_t_minus_1 - dpo_t_minus_1

    delta_ccc = ccc_t - ccc_t_minus_1

    # Gross Margins
    gm_t = (inp.sales_t - inp.cogs_t) / inp.sales_t
    gm_t_minus_1 = (inp.sales_t_minus_1 - inp.cogs_t_minus_1) / inp.sales_t_minus_1
    delta_gm = gm_t - gm_t_minus_1

    # Trigger alert if Delta CCC > +20 days while Gross Margin is static (|Delta GM| <= 0.02)
    warning_triggered = bool(delta_ccc > 20.0 and abs(delta_gm) <= 0.02)
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

    if altman.zone == "Distress":
        risk_score += 3
        findings.append(f"Altman Z'' indicates severe distress ({altman.z_double_prime:.2f} < 1.10).")
    elif altman.zone == "Grey":
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
