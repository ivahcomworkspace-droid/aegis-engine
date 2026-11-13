"""
Aegis Credit Memorandum & Visualization Service
Renders high-resolution quantitative charts using Matplotlib and compiles
institutional-grade Credit Risk Memorandums into PDF and interactive HTML.
Supports WeasyPrint compilation with automatic ReportLab fallback for cross-platform resilience.
"""

import io
import os
import base64
from typing import Dict, Optional, Tuple
import numpy as np

# Use non-interactive Agg backend for server environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from aegis_engine.schemas.credit_models import (
        ComprehensiveCreditAssessment,
        ComprehensiveCreditResponse,
        IFRS9StageInput,
        AegisInternalRiskBand,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from aegis_engine.core.vasicek import (
        calculate_vasicek_loss_distribution,
        calculate_ifrs9_ecl,
    )
except ModuleNotFoundError:
    from schemas.credit_models import (
        ComprehensiveCreditAssessment,
        ComprehensiveCreditResponse,
        IFRS9StageInput,
        AegisInternalRiskBand,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from core.vasicek import (
        calculate_vasicek_loss_distribution,
        calculate_ifrs9_ecl,
    )


def _reconcile_assessment(assessment: ComprehensiveCreditAssessment) -> Tuple[ComprehensiveCreditAssessment, str]:
    """
    Reconciles underwriting directive banner, SICR trigger thresholds,
    multi-period IFRS 9 ECL amortization schedules, and dynamic executive narrative.
    """
    m = assessment.merton
    f = assessment.forensics
    ecl = assessment.ifrs9_ecl
    dd = m.distance_to_default_physical

    # 1. Align SICR watch (DD < 2.0 sigma): ensure multi-year Stage 2 lifetime schedule
    if dd < 2.0:
        if (ecl.assigned_stage == 1 or len(ecl.period_schedules) <= 1 or not ecl.is_lifetime_ecl) and ecl.assigned_stage != 3:
            pit_pd = max(min(m.default_probability_physical, 0.999), 0.0001)
            initial_ead = ecl.period_schedules[0].ead if ecl.period_schedules else max(m.default_barrier * 1.25, 1_000_000.0)
            tenor = 5
            ead_schedule = [initial_ead * ((tenor - i) / tenor) for i in range(tenor)]

            recomputed_ecl = calculate_ifrs9_ecl(
                IFRS9StageInput(
                    current_distance_to_default=dd,
                    origination_distance_to_default=dd + 1.5,
                    current_pd=pit_pd,
                    days_past_due=30,
                    is_credit_impaired=False,
                    ead_schedule=ead_schedule,
                    effective_interest_rate=0.06,
                )
            )
            ecl = recomputed_ecl

    # 2. Align directive text: if Stage 1 is selected, adjust directive banner to state "PRIME / STABLE QUALITY"
    if ecl.assigned_stage == 1:
        rec_directive = "PRIME / STABLE QUALITY"
    elif ecl.assigned_stage == 3:
        rec_directive = assessment.recommendation
    elif dd < 2.0 or ecl.assigned_stage == 2:
        rec_directive = "MONITOR CLOSELY: Significant increase in credit risk (SICR) observed; quarterly debt covenants required."
    else:
        rec_directive = assessment.recommendation

    # 3. Dynamic Narrative Reconciler: prevent stale Stage 1 narrative when borrower is Stage 2
    total_ecl = ecl.total_ecl
    ecl_pct = ecl.ecl_percentage_of_ead
    if ecl.assigned_stage == 2:
        stage_desc = f"Classified under IFRS 9 as Stage 2 (Underperforming - SICR) with lifetime ECL allowance of ${total_ecl:,.0f} ({ecl_pct:.2f}%)."
    elif ecl.assigned_stage == 1:
        stage_desc = f"Classified under IFRS 9 as Stage 1 (Performing - 12-Month ECL) with 12-month ECL allowance of ${total_ecl:,.0f} ({ecl_pct:.2f}%)."
    else:
        stage_desc = f"Classified under IFRS 9 as Stage 3 (Credit-Impaired) with lifetime ECL allowance of ${total_ecl:,.0f} ({ecl_pct:.2f}%)."

    if not getattr(f, "is_financial_institution", False):
        forensic_clause = (
            f"Altman Z'' of {f.altman.z_double_prime:.2f} ({f.altman.zone}), and Beneish M-Score of {f.beneish.m_score:.2f} "
            f"({'MANIPULATION ALERT' if f.beneish.is_manipulator else 'CLEAN'})"
        )
    else:
        # Financial companies: the scores are shown for reference only; they are calibrated on non-financial companies
        forensic_clause = "forensic scores shown for reference only (calibrated on non-financial companies)"
        f = f.model_copy(update={"summary_findings": [
            "Financial-sector borrower: Beneish, Altman Z'' and cash-cycle scores are calibrated on non-financial "
            "companies and were excluded from the risk band."
        ]})

    reconciled_summary = (
        f"Borrower exhibits physical Distance-to-Default of {dd:.2f}σ (PD: {m.default_probability_physical:.3%}), "
        f"{forensic_clause}. {stage_desc}"
    )

    assessment = assessment.model_copy(update={
        "ifrs9_ecl": ecl,
        "forensics": f,
        "recommendation": rec_directive,
        "executive_summary": reconciled_summary,
    })

    return assessment, rec_directive


def generate_credit_charts(assessment: ComprehensiveCreditAssessment) -> Dict[str, bytes]:
    """
    Renders 3 institutional-grade credit risk charts:
    1. Merton Structural Asset & Distance-to-Default profile.
    2. Forensics Scorecard (Beneish 8-Factor, Altman Z'', CCC dynamics).
    3. Vasicek Macro Loss Distribution & IFRS 9 ECL Staging.
    Returns dictionary mapping chart names to PNG byte streams.
    """
    assessment, _ = _reconcile_assessment(assessment)
    charts: Dict[str, bytes] = {}

    plt.style.use('fast')
    dark_navy = '#0f172a'
    slate_blue = '#1e293b'
    crimson = '#dc2626'
    emerald = '#059669'
    amber = '#d97706'
    cyan = '#0284c7'

    # -------------------------------------------------------------
    # 1. Merton Structural Profile Chart
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.4), dpi=150)
    fig.patch.set_facecolor('#f8fafc')

    # Panel 1: Capital Structure & Default Barrier
    ax1.set_facecolor('#ffffff')
    categories = ['Latent Assets (V_A)', 'Default Barrier (D)', 'Market Cap (E)']
    m_out = assessment.merton
    values = [m_out.asset_value, m_out.default_barrier, m_out.asset_value - m_out.default_barrier]
    colors = [cyan, crimson, emerald]

    bars = ax1.bar(categories, values, color=colors, width=0.55, edgecolor='#334155', linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.02, f"${yval:,.1f}M", ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax1.set_title("Merton Balance Sheet Inversion", fontsize=11, fontweight='bold', color=dark_navy, pad=10)
    ax1.set_ylabel("Enterprise Value ($ Millions)", fontsize=9, color=slate_blue)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='x', labelsize=8)

    # Panel 2: Distance-to-Default & Volatility Ratio
    ax2.set_facecolor('#ffffff')
    metric_names = ['Physical DD (sigma)', 'Risk-Neutral DD', 'Asset Vol (sigma_A)', 'Equity Vol (sigma_E)']
    metric_vals = [
        m_out.distance_to_default_physical,
        m_out.distance_to_default_risk_neutral,
        m_out.asset_volatility * 100.0,
        assessment.merton.asset_volatility * 100.0 * (m_out.asset_value / (m_out.asset_value - m_out.default_barrier + 1e-6))
    ]
    # Use actual equity volatility from input if available or reconstructed
    bars2 = ax2.barh(metric_names, metric_vals, color=[emerald, cyan, amber, crimson], height=0.5, edgecolor='#334155')
    for bar in bars2:
        xval = bar.get_width()
        unit = "%" if "Vol" in bar.get_label() or bar.get_y() >= 2 else "σ"
        ax2.text(xval + 0.1, bar.get_y() + bar.get_height()/2.0, f"{xval:.2f}", ha='left', va='center', fontweight='bold', fontsize=8)

    ax2.set_title("Solvency Metrics & Volatility Inversion", fontsize=11, fontweight='bold', color=dark_navy, pad=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png', bbox_inches='tight')
    plt.close(fig)
    charts['merton_profile'] = buf1.getvalue()

    # -------------------------------------------------------------
    # 2. Forensics Scorecard Chart
    # -------------------------------------------------------------
    fig, (ax_b, ax_z, ax_ccc) = plt.subplots(1, 3, figsize=(11.0, 3.3), dpi=150)
    fig.patch.set_facecolor('#f8fafc')

    # Beneish Factors
    b_res = assessment.forensics.beneish
    factors = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'TATA', 'LVGI']
    f_vals = [b_res.dsri, b_res.gmi, b_res.aqi, b_res.sgi, b_res.depi, b_res.sgai, b_res.tata, b_res.lvgi]
    b_colors = [crimson if v > 1.25 and f != 'TATA' else (crimson if f == 'TATA' and v > 0.05 else cyan) for f, v in zip(factors, f_vals)]

    ax_b.set_facecolor('#ffffff')
    ax_b.bar(factors, f_vals, color=b_colors, width=0.6, edgecolor='#334155')
    ax_b.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_b.set_title(f"Beneish 8-Factor (M={b_res.m_score:.2f})", fontsize=10, fontweight='bold', color=dark_navy)
    ax_b.set_ylabel("Factor Multiplier", fontsize=8)
    ax_b.tick_params(axis='x', rotation=45, labelsize=8)
    ax_b.grid(axis='y', linestyle='--', alpha=0.4)

    # Altman Z'' Zone
    z_res = assessment.forensics.altman
    ax_z.set_facecolor('#ffffff')
    z_score = z_res.z_double_prime
    z_color = emerald if z_res.zone == "Safe" else (amber if z_res.zone == "Grey" else crimson)
    ax_z.barh(['Altman Z"'], [z_score], color=z_color, height=0.4, edgecolor='#334155')
    ax_z.axvline(1.10, color=crimson, linestyle='--', linewidth=1.5, label='Distress (<1.10)')
    ax_z.axvline(2.60, color=emerald, linestyle='--', linewidth=1.5, label='Safe (>2.60)')
    ax_z.set_xlim(0, max(z_score * 1.3, 4.0))
    ax_z.text(z_score + 0.1, 0, f"{z_score:.2f} ({z_res.zone})", va='center', fontweight='bold', fontsize=9)
    ax_z.set_title("Altman Z''-Score Zone", fontsize=10, fontweight='bold', color=dark_navy)
    ax_z.legend(loc='lower right', fontsize=8)
    ax_z.grid(axis='x', linestyle='--', alpha=0.4)

    # CCC Dynamics
    ccc_res = assessment.forensics.ccc
    ax_ccc.set_facecolor('#ffffff')
    ccc_cats = ['DSO', 'DIO', 'DPO', 'CCC Total']
    t_vals = [ccc_res.dso_t, ccc_res.dio_t, ccc_res.dpo_t, ccc_res.ccc_t]
    t_minus_1_vals = [ccc_res.dso_t_minus_1, ccc_res.dio_t_minus_1, ccc_res.dpo_t_minus_1, ccc_res.ccc_t_minus_1]
    x_idx = np.arange(len(ccc_cats))
    w = 0.35
    ax_ccc.bar(x_idx - w/2, t_minus_1_vals, width=w, label='Period t-1', color='#94a3b8')
    ax_ccc.bar(x_idx + w/2, t_vals, width=w, label='Period t', color=cyan if not ccc_res.warning_triggered else crimson)
    ax_ccc.set_xticks(x_idx)
    ax_ccc.set_xticklabels(ccc_cats, fontsize=8)
    ax_ccc.set_title(f"Working Capital (ΔCCC = {ccc_res.delta_ccc:+.1f}d)", fontsize=10, fontweight='bold', color=dark_navy)
    ax_ccc.set_ylabel("Days", fontsize=8)
    ax_ccc.legend(fontsize=8)
    ax_ccc.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    buf2 = io.BytesIO()
    fig.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig)
    charts['forensics_scorecard'] = buf2.getvalue()

    # -------------------------------------------------------------
    # 3. Vasicek Loss Distribution & IFRS 9 ECL Chart
    # -------------------------------------------------------------
    fig, (ax_v, ax_ecl) = plt.subplots(1, 2, figsize=(11.0, 3.3), dpi=150)
    fig.patch.set_facecolor('#f8fafc')

    # Vasicek density
    ax_v.set_facecolor('#ffffff')
    v_out = assessment.vasicek
    x_loss, density = calculate_vasicek_loss_distribution(v_out.pit_pd, v_out.asset_correlation_rho, num_points=300)
    ax_v.plot(x_loss * 100.0, density, color=slate_blue, linewidth=2.0, label='Vasicek Density f(x)')
    ax_v.fill_between(x_loss * 100.0, 0, density, color=cyan, alpha=0.2)

    # Annotate PIT PD and 99.9% VaR
    pit_pct = v_out.pit_pd * 100.0
    ax_v.axvline(pit_pct, color=emerald, linestyle='--', linewidth=1.5, label=f'PIT PD ({pit_pct:.2f}%)')

    # Find 99.9% VaR conditional PD
    reg_pct = v_out.stress_scenarios.get(list(v_out.stress_scenarios.keys())[-1], 0.05) * 100.0
    ax_v.axvline(reg_pct, color=crimson, linestyle='-', linewidth=2.0, label=f'99.9% VaR ({reg_pct:.2f}%)')

    ax_v.set_xlim(0, min(max(reg_pct * 1.5, 5.0), 100.0))
    ax_v.set_title("Vasicek Macro Credit Loss Distribution", fontsize=11, fontweight='bold', color=dark_navy)
    ax_v.set_xlabel("Loss Rate (% of EAD)", fontsize=9)
    ax_v.set_ylabel("Probability Density", fontsize=9)
    ax_v.legend(fontsize=8, loc='upper right')
    ax_v.grid(True, linestyle='--', alpha=0.4)

    # IFRS 9 Multi-Period ECL
    ax_ecl.set_facecolor('#ffffff')
    ecl_res = assessment.ifrs9_ecl
    periods = [f"Y{p.period}" for p in ecl_res.period_schedules]
    ecl_vals = [p.discounted_ecl for p in ecl_res.period_schedules]
    cum_pds = [p.cumulative_pd * 100.0 for p in ecl_res.period_schedules]

    ax_ecl.bar(periods, ecl_vals, color=amber, width=0.5, edgecolor='#334155', label='Discounted ECL ($)')
    ax_ecl.set_title(f"IFRS 9 {ecl_res.stage_name}", fontsize=11, fontweight='bold', color=dark_navy)
    ax_ecl.set_ylabel("Expected Credit Loss ($)", fontsize=9, color=amber)
    ax_ecl.grid(axis='y', linestyle='--', alpha=0.4)

    # Secondary axis for cumulative PD
    ax_ecl_sec = ax_ecl.twinx()
    ax_ecl_sec.plot(periods, cum_pds, color=crimson, marker='o', linewidth=2, label='Cum. PD (%)')
    ax_ecl_sec.set_ylabel("Cumulative PD (%)", fontsize=9, color=crimson)

    plt.tight_layout()
    buf3 = io.BytesIO()
    fig.savefig(buf3, format='png', bbox_inches='tight')
    plt.close(fig)
    charts['vasicek_loss_dist'] = buf3.getvalue()

    return charts


def generate_credit_memo_html(assessment: ComprehensiveCreditAssessment) -> str:
    """
    Renders institutional HTML memorandum for web viewing or WeasyPrint compilation.
    """
    assessment, rec_directive = _reconcile_assessment(assessment)
    charts = generate_credit_charts(assessment)
    merton_b64 = base64.b64encode(charts['merton_profile']).decode('utf-8')
    forensics_b64 = base64.b64encode(charts['forensics_scorecard']).decode('utf-8')
    vasicek_b64 = base64.b64encode(charts['vasicek_loss_dist']).decode('utf-8')

    m = assessment.merton
    f = assessment.forensics
    v = assessment.vasicek
    ecl = assessment.ifrs9_ecl
    stage_badge = f"Stage {ecl.assigned_stage}"

    band_text = (
        getattr(assessment, "internal_risk_band", None)
        or getattr(assessment, "composite_rating", None)
        or getattr(assessment, "credit_rating", "Band 4 (Acceptable)")
    )
    disclaimer_text = (
        getattr(assessment, "regulatory_disclaimer", None)
        or DEFAULT_REGULATORY_DISCLAIMER
    )

    if any(b in band_text for b in ["Band 1", "Band 2", "Band 3"]):
        rating_color = "#059669"
    elif any(b in band_text for b in ["Band 4"]):
        rating_color = "#0284c7"
    elif any(b in band_text for b in ["Band 5"]):
        rating_color = "#d97706"
    else:
        rating_color = "#dc2626"

    stage_color = "#059669" if ecl.assigned_stage == 1 else ("#d97706" if ecl.assigned_stage == 2 else "#dc2626")

    red_flags_html = "".join([f"<li>{rf}</li>" for rf in f.beneish.red_flags])
    if not red_flags_html:
        red_flags_html = "<li>No active earnings manipulation red flags identified.</li>"

    findings_html = "".join([f"<li>{fd}</li>" for fd in f.summary_findings])

    schedule_rows = "".join([
        f"<tr><td>Y{p.period}</td><td>{p.marginal_pd*100:.2f}%</td><td>{p.cumulative_pd*100:.2f}%</td>"
        f"<td>{p.survival_probability*100:.2f}%</td><td>${p.ead:,.0f}</td><td>${p.discounted_ecl:,.2f}</td></tr>"
        for p in ecl.period_schedules
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AEGIS Credit Risk Memorandum — {assessment.company_name}</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 12mm 12mm 12mm 12mm;
    @bottom-right {{
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8pt;
      color: #64748b;
    }}
    @bottom-left {{
      content: "AEGIS INTERNAL RISK MODEL // FOR SCREENING PURPOSES ONLY";
      font-size: 7.5pt;
      font-weight: bold;
      color: #64748b;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    background-color: #ffffff;
    line-height: 1.35;
    font-size: 9pt;
    margin: 0;
    padding: 0;
  }}
  .header {{
    border-bottom: 2px solid #1e293b;
    padding-bottom: 6px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .logo {{
    font-size: 18pt;
    font-weight: 900;
    letter-spacing: 1.2px;
    color: #0f172a;
  }}
  .logo span {{ color: #0284c7; }}
  .subtitle {{
    font-size: 8.5pt;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 1px;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    color: #ffffff;
    font-weight: 700;
    font-size: 8.5pt;
  }}
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    background: #f1f5f9;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
  }}
  .meta-cell strong {{
    display: block;
    font-size: 7pt;
    color: #64748b;
    text-transform: uppercase;
  }}
  .meta-cell span {{
    font-size: 10.5pt;
    font-weight: 700;
    color: #0f172a;
  }}
  h2 {{
    font-size: 11pt;
    color: #1e293b;
    border-left: 4px solid #0284c7;
    padding-left: 8px;
    margin-top: 8px;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  h2.section-3 {{
    margin-top: 4px;
  }}
  h3 {{
    font-size: 8.5pt;
    color: #334155;
    margin-top: 4px;
    margin-bottom: 2px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 4px;
    margin-bottom: 6px;
    font-size: 7.5pt;
  }}
  th, td {{
    padding: 3px 6px;
    font-size: 7.5pt;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
  }}
  th {{
    background-color: #f8fafc;
    color: #334155;
    font-weight: 700;
    border-bottom: 1.5px solid #cbd5e1;
  }}
  .chart-img {{
    width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    margin-top: 4px;
    margin-bottom: 6px;
  }}
  .recommendation-box {{
    background: #eff6ff;
    border-left: 4px solid #0284c7;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
  }}
  .page-break {{
    page-break-before: always;
  }}
  ul {{
    margin: 2px 0 6px 16px;
    padding: 0;
  }}
  li {{
    margin-bottom: 2px;
  }}
  .ifrs9-table {{
    page-break-inside: avoid;
    margin-top: 4px;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="logo">AEGIS <span>CREDIT</span></div>
    <div class="subtitle">Institutional Quantitative Credit Risk & Balance Sheet Forensics</div>
  </div>
  <div style="text-align: right;">
    <div style="font-size: 8pt; color: #64748b;">EVALUATION TIMESTAMP</div>
    <div style="font-weight: bold; font-size: 9.5pt;">{assessment.timestamp}</div>
  </div>
</div>

<div class="meta-grid">
  <div class="meta-cell">
    <strong>Borrower Entity</strong>
    <span>{assessment.company_name} {f"({assessment.ticker})" if assessment.ticker else ""}</span>
  </div>
  <div class="meta-cell">
    <strong>Internal Risk Band</strong>
    <span style="color: {rating_color}; font-weight: 700;">{band_text}</span>
  </div>
  <div class="meta-cell">
    <strong>IFRS 9 Staging</strong>
    <span style="color: {stage_color};">{stage_badge}</span>
  </div>
  <div class="meta-cell">
    <strong>Total ECL Allowance</strong>
    <span>${ecl.total_ecl:,.0f} ({ecl.ecl_percentage_of_ead:.2f}%)</span>
  </div>
</div>

<div class="recommendation-box">
  <strong style="color: #1e3a8a; text-transform: uppercase; font-size: 8.5pt;">Indicative Underwriting Stance</strong>
  <div style="font-size: 9.5pt; font-weight: 700; margin-top: 2px; color: #0f172a;">{rec_directive}</div>
  <p style="margin: 4px 0 0 0; font-size: 8.5pt; color: #334155;">{assessment.executive_summary}</p>
</div>

<h2>1. Merton Structural Asset Inversion & Distance-to-Default</h2>
<table style="width: 100%;">
  <tr>
    <th>Latent Asset Value (V_A)</th>
    <th>Asset Volatility (σ_A)</th>
    <th>Default Barrier (D)</th>
    <th>Physical DD</th>
    <th>Physical PD</th>
    <th>Credit Spread</th>
  </tr>
  <tr>
    <td>${m.asset_value:,.2f}M</td>
    <td>{m.asset_volatility*100:.2f}%</td>
    <td>${m.default_barrier:,.2f}M</td>
    <td><b>{m.distance_to_default_physical:.2f}σ</b></td>
    <td><b>{m.default_probability_physical*100:.3f}%</b></td>
    <td><b>{m.credit_spread_bps:.1f} bps</b></td>
  </tr>
</table>

<img class="chart-img" src="data:image/png;base64,{merton_b64}" alt="Merton Profile Chart"/>

<div class="page-break"></div>

<h2>2. Balance Sheet Forensics & Working Capital Stress</h2>
<table style="width: 100%;">
  <tr>
    <th>Beneish M-Score</th>
    <th>Manipulation Flag</th>
    <th>Altman Z''-Score</th>
    <th>Altman Solvency Zone</th>
    <th>Δ Cash Conversion Cycle</th>
  </tr>
  <tr>
    <td><b>{f.beneish.m_score:.2f}</b></td>
    <td style="color: {'#dc2626' if f.beneish.is_manipulator else '#059669'}; font-weight: bold;">
      {'MANIPULATOR WARNING' if f.beneish.is_manipulator else 'Clean / Safe'}
    </td>
    <td><b>{f.altman.z_double_prime:.2f}</b></td>
    <td><b>{f.altman.zone}</b></td>
    <td style="color: {'#dc2626' if f.ccc.warning_triggered else '#0f172a'}; font-weight: bold;">
      {f.ccc.delta_ccc:+.1f} days
    </td>
  </tr>
</table>

<img class="chart-img" src="data:image/png;base64,{forensics_b64}" alt="Forensics Scorecard"/>

<h3 style="font-size: 8.5pt; color: #334155; margin-top: 4px; margin-bottom: 2px;">Forensic Investigator Findings:</h3>
<ul style="font-size: 8pt; color: #1e293b;">
  {findings_html}
</ul>

<h2 class="section-3" style="margin-top: 4px;">3. Regulatory Loss Distribution (Vasicek ASRF) & IFRS 9 ECL</h2>
<table style="width: 100%; margin-top: 4px;">
  <tr>
    <th>PIT PD</th>
    <th>Asset Correlation (ρ)</th>
    <th>99.9% VaR Cond. PD</th>
    <th>Basel Capital (K)</th>
    <th>Expected Loss (EL)</th>
    <th>Unexpected Loss (UL)</th>
  </tr>
  <tr>
    <td>{v.pit_pd*100:.2f}%</td>
    <td>{v.asset_correlation_rho:.4f}</td>
    <td><b>{v.stress_scenarios.get(list(v.stress_scenarios.keys())[-1], 0)*100:.2f}%</b></td>
    <td>{v.regulatory_capital_k*100:.2f}%</td>
    <td>${v.expected_loss:,.0f}</td>
    <td>${v.unexpected_loss:,.0f}</td>
  </tr>
</table>

<img class="chart-img" src="data:image/png;base64,{vasicek_b64}" alt="Vasicek Loss Distribution"/>

<h3 style="font-size: 8.5pt; color: #334155; margin-top: 4px; margin-bottom: 2px;">IFRS 9 Amortization & Loss Provisioning Schedule:</h3>
<table class="ifrs9-table" style="page-break-inside: avoid; margin-top: 4px;">
  <thead>
    <tr>
      <th>Period</th>
      <th>Marginal PD</th>
      <th>Cumulative PD</th>
      <th>Survival Prob.</th>
      <th>EAD</th>
      <th>Discounted ECL</th>
    </tr>
  </thead>
  <tbody>
    {schedule_rows}
  </tbody>
</table>

<div class="disclaimer-box" style="margin-top: 6px; padding: 4px 8px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; font-size: 6pt; color: #64748b; line-height: 1.25; text-align: justify; page-break-inside: avoid;">
  <strong>DISCLAIMER:</strong> {disclaimer_text}
</div>

</body>
</html>
"""
    return html_content


def generate_credit_memo_pdf_reportlab(assessment: ComprehensiveCreditAssessment) -> bytes:
    """
    Institutional PDF compiler using ReportLab with embedded vector styling and Matplotlib charts.
    Ensures 100% deterministic compilation across all platforms without native Cairo dependencies.
    """
    assessment, rec_directive = _reconcile_assessment(assessment)
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    charts = generate_credit_charts(assessment)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b')
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6,
        spaceBefore=10
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    story = []

    # Header
    story.append(Paragraph("<b>AEGIS</b> <font color='#0284c7'>CREDIT</font>", title_style))
    story.append(Paragraph("INSTITUTIONAL QUANTITATIVE CREDIT RISK & FORENSICS ENGINE", subtitle_style))
    story.append(Spacer(1, 10))

    # Summary Meta Table
    m = assessment.merton
    f = assessment.forensics
    v = assessment.vasicek
    ecl = assessment.ifrs9_ecl
    stage_badge = f"Stage {ecl.assigned_stage}"

    band_text = (
        getattr(assessment, "internal_risk_band", None)
        or getattr(assessment, "composite_rating", None)
        or getattr(assessment, "credit_rating", "Band 4 (Acceptable)")
    )
    disclaimer_text = (
        getattr(assessment, "regulatory_disclaimer", None)
        or DEFAULT_REGULATORY_DISCLAIMER
    )

    meta_data = [
        [
            Paragraph("<b>Borrower:</b> " + assessment.company_name, body_style),
            Paragraph("<b>Risk Band:</b> " + band_text, body_style),
            Paragraph(f"<b>IFRS 9:</b> {stage_badge}", body_style),
            Paragraph(f"<b>Total ECL:</b> ${ecl.total_ecl:,.0f}", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[1.8*inch, 1.4*inch, 1.8*inch, 2.0*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Indicative Stance
    rec_text = f"<b>Indicative Stance:</b> {rec_directive}<br/>{assessment.executive_summary}"
    rec_data = [[Paragraph(rec_text, body_style)]]
    rec_table = Table(rec_data, colWidths=[7.0*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#0284c7')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # Section 1: Merton
    story.append(Paragraph("1. Merton Structural Inversion & Distance-to-Default", section_heading))
    merton_table_data = [
        ["Asset Value (V_A)", "Asset Vol (σ_A)", "Default Barrier (D)", "Physical DD", "Physical PD", "Spread"],
        [
            f"${m.asset_value:,.1f}M",
            f"{m.asset_volatility*100:.2f}%",
            f"${m.default_barrier:,.1f}M",
            f"{m.distance_to_default_physical:.2f}σ",
            f"{m.default_probability_physical*100:.3f}%",
            f"{m.credit_spread_bps:.1f} bps"
        ]
    ]
    t1 = Table(merton_table_data, colWidths=[1.2*inch, 1.1*inch, 1.2*inch, 1.1*inch, 1.1*inch, 1.2*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    img1 = Image(io.BytesIO(charts['merton_profile']), width=6.8*inch, height=2.1*inch)
    story.append(img1)

    story.append(PageBreak())

    # Section 2: Forensics
    story.append(Paragraph("2. Balance Sheet Forensics & Working Capital Stress", section_heading))
    forensic_table_data = [
        ["Beneish M-Score", "Manipulation Flag", "Altman Z''-Score", "Altman Zone", "Δ CCC (Days)"],
        [
            f"{f.beneish.m_score:.2f}",
            "ACTIVE FLAG" if f.beneish.is_manipulator else "Clean",
            f"{f.altman.z_double_prime:.2f}",
            f.altman.zone,
            f"{f.ccc.delta_ccc:+.1f} days"
        ]
    ]
    t2 = Table(forensic_table_data, colWidths=[1.4*inch, 1.4*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    img2 = Image(io.BytesIO(charts['forensics_scorecard']), width=6.8*inch, height=1.9*inch)
    story.append(img2)
    story.append(Spacer(1, 6))

    # Section 3: Vasicek & ECL
    story.append(Paragraph("3. Regulatory Loss Distribution & IFRS 9 Multi-Period ECL", section_heading))
    img3 = Image(io.BytesIO(charts['vasicek_loss_dist']), width=6.8*inch, height=1.9*inch)
    story.append(img3)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<font size='5' color='#64748b'><b>DISCLAIMER:</b> {disclaimer_text}</font>", body_style))

    doc.build(story)
    return pdf_buffer.getvalue()


def generate_credit_memo_pdf(assessment: ComprehensiveCreditAssessment) -> bytes:
    """
    Compiles credit memo to PDF. Tries WeasyPrint first; if not available, uses ReportLab compiler.
    """
    try:
        import weasyprint
        html_text = generate_credit_memo_html(assessment)
        pdf_bytes = weasyprint.HTML(string=html_text).write_pdf()
        return pdf_bytes
    except Exception:
        # Seamlessly fallback to ReportLab
        return generate_credit_memo_pdf_reportlab(assessment)


class CreditMemoGenerator:
    """High-level facade for credit memo artifact generation."""

    @classmethod
    def create_memo_pdf(cls, assessment: ComprehensiveCreditAssessment) -> bytes:
        return generate_credit_memo_pdf(assessment)

    @classmethod
    def create_memo_html(cls, assessment: ComprehensiveCreditAssessment) -> str:
        return generate_credit_memo_html(assessment)

    @classmethod
    def save_memo(cls, assessment: ComprehensiveCreditAssessment, output_pdf_path: str):
        pdf_bytes = cls.create_memo_pdf(assessment)
        with open(output_pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        return output_pdf_path
