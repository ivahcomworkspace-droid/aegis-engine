"""
Aegis credit memorandum.

Renders a comprehensive assessment (the /api/v1/credit/comprehensive response) as a print-ready
memo: HTML compiled to PDF by WeasyPrint, with a ReportLab fallback. Every figure comes from the
assessment as the engine returned it and uses the dashboard's formats (services/formatting.py), so
the memo and the screen agree. Charts are vector (SVG) so they stay sharp in print.
"""

import base64
import html
import io
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
from matplotlib import font_manager
from matplotlib import ticker as mticker
from matplotlib.figure import Figure

try:
    from aegis_engine.schemas.credit_models import (
        ComprehensiveCreditAssessment,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from aegis_engine.core.vasicek import calculate_vasicek_conditional_pd, SICR_WATCHLIST_DD
    from aegis_engine.services.formatting import format_money_m, format_pd, format_pct
except ModuleNotFoundError:
    from schemas.credit_models import (
        ComprehensiveCreditAssessment,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from core.vasicek import calculate_vasicek_conditional_pd, SICR_WATCHLIST_DD
    from services.formatting import format_money_m, format_pd, format_pct


# --- Palette (UrbanHome brand colours, print-weight status colours) ------------------------------

NAVY = "#141B47"
PRIMARY = "#4667D5"
PRIMARY_TINT = "#E8EDFB"
INK = "#2B3245"
MUTED = "#6B7385"
HAIRLINE = "#DDE2EC"
PANEL = "#F5F7FB"
GOOD, GOOD_TINT = "#1F8A5B", "#E3F4EC"
WARN, WARN_TINT = "#B7700B", "#FBF0DC"
BAD, BAD_TINT = "#C2362F", "#FBE4E2"
TONES = {"good": (GOOD, GOOD_TINT), "info": (PRIMARY, PRIMARY_TINT), "warn": (WARN, WARN_TINT), "bad": (BAD, BAD_TINT)}

MARKET_INPUTS = ("equity_value", "equity_volatility", "short_term_debt", "long_term_debt", "risk_free_rate")
SOLVERS = {"hybr": "Powell hybrid", "trf_fallback": "trust-region fallback", "zero_debt_exact": "exact (no debt)"}


def _chart_font() -> str:
    """Lato when installed (the memo's text face), else a metric-compatible fallback."""
    for family in ("Lato", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


# Set once at import (never changed afterwards, so concurrent memo renders can't interfere)
matplotlib.rcParams.update({
    "font.family": _chart_font(),
    "font.size": 7.0,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 0.6,
    "axes.labelcolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "grid.color": "#ECEFF5",
    "grid.linewidth": 0.5,
    "svg.fonttype": "path",
    "svg.hashsalt": "aegis-memo",
})


# --- Small helpers -------------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or key access: the assessment may hold models or plain dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sigma(value: float) -> str:
    return f"{value:.2f}σ"


def _band_parts(band_text: str) -> Tuple[str, str, int]:
    """'Band 2 (Low Credit Risk)' -> ('Band 2', 'Low Credit Risk', 2)."""
    label, _, rest = band_text.partition(" (")
    name = rest.rstrip(")") if rest else ""
    digits = "".join(ch for ch in label if ch.isdigit())
    return label, name, int(digits) if digits else 4


def _band_tone(number: int) -> str:
    """Bands 1-4 acceptable, 5 elevated, 6-7 high: the dashboard's colours."""
    return "good" if number <= 4 else "warn" if number == 5 else "bad"


def _stage_tone(stage: int) -> str:
    return {1: "good", 2: "warn"}.get(stage, "bad")


def _stance_parts(stance: str) -> Tuple[str, str]:
    title, _, text = stance.partition(" — ")
    return title.strip(), text.strip()


def _pretty_rationale(text: str) -> str:
    return text.replace("sigma", "σ").replace("Delta DD", "ΔDD")


def _prepared(timestamp: str) -> Tuple[str, str]:
    """('14 November 2026, 06:59 UTC', '20261114-0659') from the assessment's UTC timestamp."""
    try:
        moment = datetime.strptime(timestamp.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        return f"{moment.day} {moment:%B %Y, %H:%M} UTC", f"{moment:%Y%m%d-%H%M}"
    except (ValueError, AttributeError):
        return str(timestamp), "00000000-0000"


# Kept for callers that pass amounts of unknown unit (it guesses: below 1.0 means $ millions).
# The memo itself knows its units and uses format_money_m, the dashboard's format.
def format_currency(val: Optional[float], is_millions: Optional[bool] = None) -> str:
    """
    Formats currency amounts with adaptive precision, preserving small values accurately
    without truncating sub-$10k figures (such as $5.5k) to $0.
    """
    if val is None:
        return "$0"
    v = float(val)
    if abs(v) < 1e-9:
        return "$0"

    # Auto-detect whether value is expressed in millions ($M) or raw dollars:
    # In the Aegis engine, values with 0.0 < abs(v) < 1.0 (such as 0.0055) represent $M ($5.5k).
    if is_millions is True or (is_millions is None and 0.0 < abs(v) < 1.0):
        dollars = v * 1_000_000.0
    else:
        dollars = v

    abs_dollars = abs(dollars)
    if abs_dollars >= 1e12:
        return f"${dollars / 1e12:,.2f}T"
    elif abs_dollars >= 1e9:
        return f"${dollars / 1e9:,.2f}B"
    elif abs_dollars >= 1e6:
        return f"${dollars / 1e6:,.2f}M"
    elif abs_dollars >= 1e3:
        k_val = dollars / 1e3
        return f"${k_val:.1f}k"
    else:
        return f"${dollars:,.2f}"


# --- Figures (one place, shared by the HTML and the ReportLab renderers) ------------------------

def memo_facts(assessment: ComprehensiveCreditAssessment) -> Dict[str, Any]:
    """Every figure the memo prints, formatted once."""
    a = assessment
    m, f, ecl = a.merton, a.forensics, a.ifrs9_ecl
    v = a.vasicek or a.regulatory_capital
    inputs: Dict[str, Any] = dict(a.inputs or {})
    sources = {k: (s.model_dump() if hasattr(s, "model_dump") else dict(s)) for k, s in (a.input_sources or {}).items()}

    equity = inputs.get("equity_value")
    eq_vol = inputs.get("equity_volatility")
    std = inputs.get("short_term_debt")
    ltd = inputs.get("long_term_debt")
    rate = inputs.get("risk_free_rate")
    ead = inputs.get("loan_exposure", ecl.period_schedules[0].ead if ecl.period_schedules else 0.0)

    band_label, band_name, band_no = _band_parts(a.internal_risk_band)
    stance_title, stance_text = _stance_parts(a.indicative_stance)
    prepared, ref_stamp = _prepared(a.timestamp)
    ticker = (a.ticker or "").strip().upper()

    def source(key: str, fallback: str) -> Tuple[str, str]:
        note = sources.get(key) or {}
        return note.get("source") or fallback, note.get("as_of") or "—"

    fallback_source = "Supplied with the request"
    market = [str((sources.get(k) or {}).get("source", "")) for k in MARKET_INPUTS]
    if any(word in text for text in market for word in ("SEC", "Treasury", "Yahoo")):
        basis = "Live public data: SEC EDGAR filings, US Treasury yields and market prices"
        if any("Entered on the dashboard" in text for text in market):
            basis += ", with some inputs entered by hand"
    elif all("Sample" in text for text in market):
        basis = "Sample scenario (illustrative company)"
    elif any(market):
        basis = "Inputs entered on the dashboard"
    else:
        basis = "Inputs supplied with the request"

    is_fi = bool(_get(f, "is_financial_institution", False))
    stressed = [label for flag, label in (
        ("simulate_beneish_manipulation", "simulated earnings manipulation"),
        ("simulate_working_capital_stress", "simulated working-capital stress"),
    ) if inputs.get(flag)]

    scenarios = list((v.stress_scenarios or {}).items())
    cond_99 = next((p for name, p in scenarios if "99%" in name), None)
    cond_999 = scenarios[-1][1] if scenarios else v.conditional_pd

    leverage = m.leverage_ratio if m.leverage_ratio else (m.default_barrier / m.asset_value if m.asset_value else 0.0)
    coverage = (m.asset_value / m.default_barrier) if m.default_barrier else math.inf
    dd_rn_differs = abs(m.distance_to_default_risk_neutral - m.distance_to_default_physical) >= 0.005

    findings: List[str] = []
    findings.append(
        f"{a.internal_risk_band}. Distance to default of {_sigma(m.distance_to_default_physical)}: "
        f"the market-implied asset value of "
        f"{format_money_m(m.asset_value)} covers the {format_money_m(m.default_barrier)} default point "
        + (f"{coverage:.1f} times" if math.isfinite(coverage) else "with no debt outstanding")
        + f", a one-year default probability of {format_pd(m.default_probability_physical)}."
    )
    if is_fi:
        findings.append(
            "Financial-sector borrower: Beneish, Altman Z'' and cash-cycle scores are calibrated on non-financial "
            "companies and were excluded from the risk band."
        )
    else:
        findings.append(
            f"Accounting screens: Beneish M-Score {f.beneish.m_score:.2f} "
            f"({'above' if f.beneish.is_manipulator else 'below'} the -1.78 threshold), Altman Z'' "
            f"{f.altman.z_double_prime:.2f} ({f.altman.zone.title()} zone), cash cycle {f.ccc.ccc_t:.1f} days "
            f"({f.ccc.delta_ccc:+.1f} days on the prior year)."
        )
    findings.append(
        f"IFRS 9 {ecl.stage_name}: {'lifetime' if ecl.is_lifetime_ecl else '12-month'} expected credit loss of "
        f"{format_money_m(ecl.total_ecl)} on {format_money_m(ead)} of exposure ({ecl.ecl_percentage_of_ead:.3f}%)."
    )
    findings.append(
        f"Basel IRB capital at 99.9%: conditional PD {format_pd(cond_999)}, capital requirement "
        f"{format_pct(v.regulatory_capital_k, 2)} of exposure ({format_money_m(v.unexpected_loss)} unexpected loss)."
    )
    summary = a.executive_summary or ""
    if "Risk Reconciliation:" in summary:
        findings.append(summary.split("Risk Reconciliation:", 1)[1].strip())
    if stressed:
        findings.append(f"Scenario overrides active ({' and '.join(stressed)}): forensic figures reflect the stressed statements.")

    mu = inputs.get("asset_drift_mu")
    input_rows = []
    if equity is not None:
        input_rows += [
            ("Market value of equity", "E", format_money_m(equity), *source("equity_value", fallback_source)),
            ("Equity volatility", "σE", format_pct(eq_vol, 1), *source("equity_volatility", fallback_source)),
            ("Short-term debt", "STD", format_money_m(std), *source("short_term_debt", fallback_source)),
            ("Long-term debt", "LTD", format_money_m(ltd), *source("long_term_debt", fallback_source)),
        ]
    input_rows.append(("Default point", "D = STD + ½·LTD", format_money_m(m.default_barrier), "Derived (KMV convention)", "—"))
    if rate is not None:
        input_rows.append(("Risk-free rate (continuous)", "r", format_pct(rate, 2), *source("risk_free_rate", fallback_source)))
    if mu is not None:
        input_rows.append(("Asset drift", "μ", format_pct(mu, 2), "Set equal to r" if rate is not None and abs(mu - rate) < 1e-12 else "Model assumption", "—"))
    if inputs.get("maturity_years") is not None:
        input_rows.append(("Horizon · loss given default", "T · LGD",
                           f"{inputs['maturity_years']:.1f} yr · {format_pct(inputs.get('loss_given_default', 0.45), 0)}",
                           "Model assumption", "—"))
    input_rows.append(("Exposure at default", "EAD",
                       f"{format_money_m(ead)} · {inputs.get('loan_tenor_years', len(ecl.period_schedules) or 1)} yr amortising",
                       *source("loan_exposure", "Facility assumption")))
    if inputs.get("effective_interest_rate") is not None:
        input_rows.append(("Effective interest rate", "EIR", format_pct(inputs["effective_interest_rate"], 2), "ECL discounting assumption", "—"))

    return {
        "company": a.company_name,
        "ticker": ticker,
        "prepared": prepared,
        "reference": f"AEG-{ticker or 'NA'}-{ref_stamp}",
        "basis": basis,
        "forensic_basis": a.forensic_basis or ("Sample financial statements (illustrative)" if "Sample" in basis else "Financial statements supplied with the request"),
        "band_label": band_label,
        "band_name": band_name,
        "band_tone": _band_tone(band_no),
        "stance_title": stance_title,
        "stance_text": stance_text,
        "stage": ecl.assigned_stage,
        "stage_name": ecl.stage_name,
        "stage_tone": _stage_tone(ecl.assigned_stage),
        "stage_rationale": _pretty_rationale(ecl.stage_rationale),
        "kpis": [
            ("Distance to default", _sigma(m.distance_to_default_physical), "physical, 1-year"),
            ("Probability of default", format_pd(m.default_probability_physical), "physical, 1-year"),
            ("Credit spread", f"{m.credit_spread_bps:.1f} bps", "risk-neutral"),
            ("ECL allowance", format_money_m(ecl.total_ecl), f"{ecl.ecl_percentage_of_ead:.3f}% of EAD"),
            ("Asset value", format_money_m(m.asset_value), f"σV {format_pct(m.asset_volatility, 1)}"),
            ("Leverage", format_pct(leverage, 1), "default point ÷ assets"),
        ],
        "findings": findings,
        "input_rows": input_rows,
        "merton_rows": [
            ("Asset value", "V", format_money_m(m.asset_value)),
            ("Asset volatility", "σV", format_pct(m.asset_volatility, 1)),
            ("Default point", "D", format_money_m(m.default_barrier)),
            ("Leverage", "D ÷ V", format_pct(leverage, 1)),
            ("Distance to default (physical)", "DD", _sigma(m.distance_to_default_physical)),
            ("Probability of default (physical)", "N(−DD)", format_pd(m.default_probability_physical)),
        ] + ([
            ("Distance to default (risk-neutral)", "DD*", _sigma(m.distance_to_default_risk_neutral)),
            ("Probability of default (risk-neutral)", "N(−DD*)", format_pd(m.default_probability_risk_neutral)),
        ] if dd_rn_differs else []) + [
            ("Credit spread (risk-neutral)", "s", f"{m.credit_spread_bps:.1f} bps"),
            ("Solver", "", ("Converged" if m.solver_converged else "Did not converge")
             + (f" · {SOLVERS.get(m.solver_used, m.solver_used)}" if m.solver_used else "")
             + (f" · residual {m.residual_norm:.1e}" if m.residual_norm is not None else "")),
        ],
        "dd_rn_same": not dd_rn_differs,
        "is_fi": is_fi,
        "beneish": f.beneish,
        "altman": f.altman,
        "ccc": f.ccc,
        "composite": f.composite_forensic_risk,
        "forensic_notes": _forensic_notes(f, is_fi),
        "vasicek_rows": [
            ("PIT PD (floored at 0.01%)", format_pd(v.pit_pd)),
            ("Asset correlation ρ", f"{v.asset_correlation_rho:.4f}"),
            ("Conditional PD at 99%", format_pd(cond_99) if cond_99 is not None else "—"),
            ("Conditional PD at 99.9%", format_pd(cond_999)),
            ("Capital requirement K", f"{format_pct(v.regulatory_capital_k, 2)} of EAD"),
            ("Expected loss", format_money_m(v.expected_loss)),
            ("Unexpected loss (capital)", format_money_m(v.unexpected_loss)),
            ("Exposure at default", format_money_m(ead)),
        ],
        "ecl_rows": [
            (f"Year {p.period}", format_pd(p.marginal_pd), format_pd(p.cumulative_pd), format_pct(p.survival_probability, 2),
             format_money_m(p.ead), format_pct(p.lgd, 0), f"{p.discount_factor:.4f}", format_money_m(p.discounted_ecl))
            for p in ecl.period_schedules
        ],
        "ecl_total": format_money_m(ecl.total_ecl),
        "ecl_pct": f"{ecl.ecl_percentage_of_ead:.3f}%",
        "disclaimer": a.regulatory_disclaimer or DEFAULT_REGULATORY_DISCLAIMER,
        "summary": summary,
    }


def _forensic_notes(f: Any, is_fi: bool) -> List[str]:
    notes: List[str] = []
    for item in list(f.summary_findings or []) + list(f.beneish.red_flags or []) + list(f.beneish.data_quality_flags or []):
        if item and item not in notes:
            notes.append(item)
    if f.ccc.alert_message and f.ccc.alert_message not in notes:
        notes.append(f.ccc.alert_message)
    if is_fi:
        notes = [n for n in notes if "built for non-financial" not in n]
    return notes


# --- Charts --------------------------------------------------------------------------------------

def _encode(fig: Figure, fmt: str) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=220 if fmt == "png" else None, transparent=(fmt == "svg"))
    return buffer.getvalue()


def _capital_structure_chart(a: ComprehensiveCreditAssessment, fmt: str) -> bytes:
    m = a.merton
    inputs = a.inputs or {}
    rows = [("Asset value V", m.asset_value, PRIMARY)]
    if inputs.get("equity_value") is not None:
        rows.append(("Equity value E", inputs["equity_value"], "#7F96E4"))
        rows.append(("Debt STD + LTD", inputs.get("short_term_debt", 0.0) + inputs.get("long_term_debt", 0.0), "#A9B1C4"))
    rows.append(("Default point D", m.default_barrier, BAD))
    rows = rows[::-1]

    fig = Figure(figsize=(3.4, 1.75))
    ax = fig.add_axes([0.25, 0.14, 0.62, 0.8])
    labels = [r[0] for r in rows]
    values = [max(r[1], 0.0) for r in rows]
    bars = ax.barh(labels, values, color=[r[2] for r in rows], height=0.56)
    top = max(values) if values else 1.0
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + top * 0.015, bar.get_y() + bar.get_height() / 2, format_money_m(value),
                va="center", ha="left", fontsize=6.8, color=INK, fontweight="bold")
    ax.set_xlim(0, top * 1.22)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_money_m(x) if x else "0"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=6.8, labelcolor=INK)
    ax.spines["left"].set_visible(False)
    return _encode(fig, fmt)


def _distance_to_default_chart(a: ComprehensiveCreditAssessment, fmt: str) -> bytes:
    m = a.merton
    dd = m.distance_to_default_physical
    lo = min(-4.5, -dd - 0.9)
    x = np.linspace(lo, 4.0, 600)
    density = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    fig = Figure(figsize=(3.4, 1.75))
    ax = fig.add_axes([0.1, 0.22, 0.86, 0.73])
    ax.plot(x, density, color=PRIMARY, linewidth=1.2)
    ax.fill_between(x, 0, density, color=PRIMARY, alpha=0.10)
    tail = x <= -dd
    ax.fill_between(x[tail], 0, density[tail], color=BAD, alpha=0.55)
    ax.axvline(-dd, color=BAD, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(-dd, 0.41, f"  −DD = −{dd:.2f}σ\n  PD {format_pd(m.default_probability_physical)}",
            color=BAD, fontsize=6.6, va="top", ha="left" if -dd < 1.5 else "right", fontweight="bold")
    ax.set_ylim(0, 0.43)
    ax.set_xlim(lo, 4.0)
    ticks = [t for t in range(math.ceil(lo), 5)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}σ" for t in ticks])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("Standardised asset return at the horizon", fontsize=6.5)
    return _encode(fig, fmt)


def _vasicek_stress_chart(a: ComprehensiveCreditAssessment, fmt: str) -> bytes:
    v = a.vasicek or a.regulatory_capital
    z = np.linspace(-3.0, 3.6, 300)
    cond = np.array([calculate_vasicek_conditional_pd(v.pit_pd, v.asset_correlation_rho, zz) for zz in z]) * 100.0

    fig = Figure(figsize=(3.4, 1.75))
    ax = fig.add_axes([0.13, 0.22, 0.83, 0.73])
    ax.plot(z, cond, color=PRIMARY, linewidth=1.2)
    ax.fill_between(z, 0, cond, color=PRIMARY, alpha=0.08)
    # The engine's own scenario figures (the ones in the table and on the dashboard), at their Z
    scenarios = list((v.stress_scenarios or {}).items())
    marks = []
    for name, value in scenarios:
        found = re.search(r"Z=\+?(-?[\d.]+)", name)
        if not found:
            continue
        if name.startswith("Base"):
            marks.append((float(found.group(1)), "Base", value, MUTED))
        elif "99.9%" in name:
            marks.append((float(found.group(1)), "99.9%", value, BAD))
        elif "99%" in name:
            marks.append((float(found.group(1)), "99%", value, WARN))
    for z_mark, label, value, colour in marks:
        ax.plot([z_mark], [value * 100.0], marker="o", markersize=3.2, color=colour)
        ax.annotate(f"{label}: {format_pd(value)}", (z_mark, value * 100.0), textcoords="offset points",
                    xytext=(-4, 6), ha="right", fontsize=6.4, color=colour, fontweight="bold")
    ax.set_xlim(-3.0, 3.6)
    ax.set_ylim(0, max(cond.max() * 1.18, 1e-6))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2g}%"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("Systematic downturn Z (standard deviations)", fontsize=6.5)
    return _encode(fig, fmt)


def generate_credit_charts(assessment: ComprehensiveCreditAssessment, fmt: str = "svg") -> Dict[str, bytes]:
    """The memo's charts as SVG (or PNG for the ReportLab fallback), keyed by name."""
    return {
        "capital_structure": _capital_structure_chart(assessment, fmt),
        "distance_to_default": _distance_to_default_chart(assessment, fmt),
        "vasicek_stress": _vasicek_stress_chart(assessment, fmt),
    }


# --- HTML memo -----------------------------------------------------------------------------------

MEMO_CSS = """
@page {
  size: A4;
  margin: 17mm 15mm 17mm 15mm;
  @top-left { content: string(borrower); font-size: 7pt; color: #6B7385; }
  @top-right { content: "Credit Risk Memorandum"; font-size: 7pt; color: #6B7385; }
  @bottom-left { content: "Aegis Credit Risk Engine · Model output for screening only: not a credit rating or investment advice"; font-size: 6.5pt; color: #8A91A3; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: 7pt; color: #6B7385; }
}
@page :first {
  @top-left { content: none; }
  @top-right { content: none; }
}
* { box-sizing: border-box; }
html { font-family: "Lato", "Liberation Sans", "DejaVu Sans", Arial, sans-serif; font-size: 8.4pt; color: #2B3245; line-height: 1.42; }
body { margin: 0; }
.masthead { display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 7pt; border-bottom: 2.2pt solid #141B47; }
.wordmark { font-size: 17pt; font-weight: 900; letter-spacing: 3.5pt; color: #141B47; line-height: 1; }
.wordmark span { color: #4667D5; }
.doctype { margin-top: 3pt; font-size: 7.4pt; font-weight: 700; letter-spacing: 1.6pt; text-transform: uppercase; color: #4667D5; }
.meta { text-align: right; font-size: 7.2pt; color: #6B7385; line-height: 1.55; }
.meta b { color: #2B3245; font-weight: 700; }
.borrower { string-set: borrower attr(data-run); margin: 11pt 0 1pt 0; font-size: 19pt; font-weight: 900; color: #141B47; line-height: 1.15; }
.ticker { display: inline-block; vertical-align: middle; margin-left: 6pt; padding: 1.5pt 6pt; border-radius: 3pt; background: #E8EDFB; color: #4667D5; font-size: 8.5pt; font-weight: 700; letter-spacing: 0.6pt; }
.basis { font-size: 7.6pt; color: #6B7385; margin-bottom: 9pt; }
.summary { display: flex; gap: 8pt; margin-bottom: 8pt; }
.verdict { flex: 0 0 36%; border-radius: 5pt; padding: 9pt 11pt; color: #fff; }
.verdict .k { font-size: 6.8pt; font-weight: 700; letter-spacing: 1.2pt; text-transform: uppercase; opacity: 0.85; }
.verdict .band { font-size: 20pt; font-weight: 900; line-height: 1.05; margin-top: 3pt; }
.verdict .band-name { font-size: 9pt; font-weight: 700; margin-top: 1pt; }
.verdict .stage { margin-top: 7pt; padding-top: 6pt; border-top: 0.6pt solid rgba(255,255,255,0.45); font-size: 7.6pt; }
.verdict .stage b { font-weight: 900; }
.kpis { flex: 1; display: flex; flex-wrap: wrap; border: 0.6pt solid #DDE2EC; border-radius: 5pt; }
.kpi { width: 33.333%; padding: 6.5pt 9pt; border-right: 0.6pt solid #DDE2EC; border-bottom: 0.6pt solid #DDE2EC; }
.kpi:nth-child(3n) { border-right: none; }
.kpi:nth-child(n+4) { border-bottom: none; }
.kpi .k { font-size: 6.6pt; font-weight: 700; letter-spacing: 0.5pt; text-transform: uppercase; color: #6B7385; }
.kpi .v { font-size: 12.5pt; font-weight: 900; color: #141B47; margin-top: 1pt; white-space: nowrap; }
.kpi .s { font-size: 6.6pt; color: #6B7385; }
.stance { border-left: 3pt solid #4667D5; background: #F5F7FB; padding: 6.5pt 10pt; margin-bottom: 7pt; border-radius: 0 4pt 4pt 0; }
.stance { font-size: 8.2pt; color: #2B3245; }
.stance .t { font-weight: 900; color: #141B47; letter-spacing: 0.3pt; }
h2 { font-size: 10pt; font-weight: 900; color: #141B47; margin: 12pt 0 5pt 0; padding-bottom: 3pt; border-bottom: 0.8pt solid #DDE2EC; break-after: avoid; letter-spacing: 0.2pt; }
h2 .n { display: inline-block; width: 15pt; color: #4667D5; }
h3 { font-size: 8pt; font-weight: 900; color: #141B47; margin: 7pt 0 3pt 0; break-after: avoid; text-transform: uppercase; letter-spacing: 0.6pt; }
p { margin: 0 0 5pt 0; }
ul.findings { margin: 0 0 4pt 0; padding-left: 11pt; }
ul.findings li { margin-bottom: 2.5pt; }
ul.findings li::marker { color: #4667D5; }
table { width: 100%; border-collapse: collapse; font-size: 7.7pt; font-variant-numeric: tabular-nums; }
th { text-align: left; font-size: 6.6pt; font-weight: 700; letter-spacing: 0.5pt; text-transform: uppercase; color: #6B7385; padding: 3.5pt 5pt; border-bottom: 0.9pt solid #141B47; }
td { padding: 3.2pt 5pt; border-bottom: 0.5pt solid #E6E9F0; vertical-align: top; }
td.num, th.num { text-align: right; white-space: nowrap; }
td.sym { color: #6B7385; white-space: nowrap; }
td.src { color: #4A5266; }
tr.total td { font-weight: 900; color: #141B47; border-top: 0.9pt solid #141B47; border-bottom: none; }
.kv { display: flex; gap: 12pt; }
.kv table { flex: 1; }
.kv td:first-child, table.pairs td:first-child, table.pairs td:nth-child(5) { color: #4A5266; }
table.pairs td.num { font-weight: 700; color: #141B47; }
table.pairs td.gap { width: 14pt; border-bottom: none; }
table.pairs { break-inside: avoid; }
.kv td.num { font-weight: 700; color: #141B47; }
.figures { display: flex; gap: 10pt; margin-top: 6pt; break-inside: avoid; }
.figure { flex: 1; border: 0.6pt solid #DDE2EC; border-radius: 4pt; padding: 5pt 6pt 3pt 6pt; }
.figure .cap { font-size: 6.9pt; font-weight: 700; color: #141B47; margin-bottom: 1pt; }
.figure .cap span { font-weight: 400; color: #6B7385; }
.figure img { width: 100%; display: block; }
.block { break-inside: avoid; }
.scores { display: flex; gap: 8pt; margin: 4pt 0 6pt 0; break-inside: avoid; }
.score { flex: 1; border: 0.6pt solid #DDE2EC; border-radius: 4pt; padding: 6pt 8pt 7pt 8pt; }
.score .k { font-size: 6.6pt; font-weight: 700; letter-spacing: 0.5pt; text-transform: uppercase; color: #6B7385; }
.score .v { font-size: 13pt; font-weight: 900; color: #141B47; }
.pill { display: inline-block; margin-left: 4pt; padding: 0.5pt 5pt; border-radius: 6pt; font-size: 6.6pt; font-weight: 700; vertical-align: 2pt; }
.score .rule { font-size: 6.5pt; color: #6B7385; margin-top: 1pt; }
.scale { position: relative; height: 5pt; margin-top: 5pt; border-radius: 2.5pt; overflow: hidden; }
.scale .z { position: absolute; top: 0; bottom: 0; }
.scale-wrap { position: relative; }
.marker { position: absolute; top: 3pt; width: 2pt; height: 9pt; margin-left: -1pt; background: #141B47; border-radius: 1pt; }
.muted-scores .score .v { color: #8A91A3; }
.note { border: 0.6pt solid #E9D5A8; background: #FBF5E8; color: #6B4E0F; border-radius: 4pt; padding: 5pt 8pt; margin: 4pt 0 6pt 0; font-size: 7.6pt; }
.stagebox { display: flex; gap: 8pt; align-items: stretch; margin-bottom: 5pt; }
.stagebadge { flex: 0 0 auto; border-radius: 4pt; padding: 5pt 9pt; font-weight: 900; font-size: 9pt; }
.stagebox p { margin: 0; flex: 1; font-size: 7.7pt; color: #4A5266; align-self: center; }
.method { font-size: 7pt; color: #4A5266; }
.method ul { margin: 0; padding-left: 11pt; }
.method li { margin-bottom: 2pt; }
.disclaimer { margin-top: 8pt; padding: 6pt 8pt; border-radius: 4pt; background: #F5F7FB; color: #6B7385; font-size: 6.6pt; line-height: 1.4; break-inside: avoid; }
"""


def _tone_style(tone: str) -> Tuple[str, str]:
    return TONES.get(tone, TONES["info"])


def _pill(text: str, tone: str) -> str:
    fg, bg = _tone_style(tone)
    return f'<span class="pill" style="color:{fg};background:{bg}">{_esc(text)}</span>'


def _scale(value: float, lo: float, hi: float, zones: List[Tuple[float, float, str]]) -> str:
    """A horizontal scale with coloured zones and a marker at the (clamped) value."""
    span = hi - lo
    parts = []
    for start, end, tone in zones:
        left = (max(start, lo) - lo) / span * 100
        width = (min(end, hi) - max(start, lo)) / span * 100
        parts.append(f'<div class="z" style="left:{left:.2f}%;width:{width:.2f}%;background:{_tone_style(tone)[1]}"></div>')
    position = (min(max(value, lo), hi) - lo) / span * 100
    return (f'<div class="scale-wrap"><div class="scale">{"".join(parts)}</div>'
            f'<div class="marker" style="left:{position:.2f}%"></div></div>')


NUM = ' class="num"'


def _img(data: bytes, alt: str) -> str:
    return f'<img alt="{_esc(alt)}" src="data:image/svg+xml;base64,{base64.b64encode(data).decode("ascii")}"/>'


def generate_credit_memo_html(assessment: ComprehensiveCreditAssessment) -> str:
    """The memo as standalone HTML (also the source WeasyPrint compiles to PDF)."""
    fx = memo_facts(assessment)
    charts = generate_credit_charts(assessment, "svg")
    band_fg, _ = _tone_style(fx["band_tone"])
    stage_fg, stage_bg = _tone_style(fx["stage_tone"])
    b, z, c = fx["beneish"], fx["altman"], fx["ccc"]

    kpis = "".join(
        f'<div class="kpi"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div><div class="s">{_esc(s)}</div></div>'
        for k, v, s in fx["kpis"]
    )
    findings = "".join(f"<li>{_esc(item)}</li>" for item in fx["findings"])
    input_rows = "".join(
        f'<tr><td>{_esc(name)}</td><td class="sym">{_esc(sym)}</td><td class="num">{_esc(val)}</td>'
        f'<td class="src">{_esc(src)}</td><td class="num">{_esc(asof)}</td></tr>'
        for name, sym, val, src, asof in fx["input_rows"]
    )
    half = (len(fx["merton_rows"]) + 1) // 2
    left, right = fx["merton_rows"][:half], fx["merton_rows"][half:]
    right = right + [("", "", "")] * (len(left) - len(right))
    merton_table = '<table class="pairs">' + "".join(
        f'<tr><td>{_esc(n1)}</td><td class="sym">{_esc(s1)}</td><td class="num">{_esc(v1)}</td>'
        f'<td class="gap"></td><td>{_esc(n2)}</td><td class="sym">{_esc(s2)}</td><td class="num">{_esc(v2)}</td></tr>'
        for (n1, s1, v1), (n2, s2, v2) in zip(left, right)
    ) + "</table>"
    vasicek_table = "<table>" + "".join(
        f'<tr><td>{_esc(k)}</td><td class="num">{_esc(v)}</td></tr>' for k, v in fx["vasicek_rows"]
    ) + "</table>"

    # Forensic scorecards
    m_tone = "bad" if b.is_manipulator else "good"
    z_zone = (z.zone or "").title()
    z_tone = {"Safe": "good", "Grey": "warn", "Distress": "bad"}.get(z_zone, "info")
    c_tone = "bad" if c.warning_triggered else ("warn" if c.delta_ccc > 10.0 else "good")
    c_label = "Alert" if c.warning_triggered else ("Lengthening" if c.delta_ccc > 10.0 else "Stable")
    if fx["is_fi"]:
        scores = """
    <div class="scores muted-scores">
      <div class="score"><div class="k">Beneish M-Score</div><div class="v">n/a</div><div class="rule">Not applied to financial companies</div></div>
      <div class="score"><div class="k">Altman Z″-Score</div><div class="v">n/a</div><div class="rule">Not applied to financial companies</div></div>
      <div class="score"><div class="k">Cash conversion cycle</div><div class="v">n/a</div><div class="rule">Not applied to financial companies</div></div>
    </div>"""
    else:
        scores = f"""
    <div class="scores">
      <div class="score"><div class="k">Beneish M-Score</div>
        <div class="v">{b.m_score:.2f}{_pill('Manipulation flag' if b.is_manipulator else 'Clean', m_tone)}</div>
        <div class="rule">Above -1.78 flags likely earnings manipulation</div>
        {_scale(b.m_score, -4.0, 0.0, [(-4.0, -1.78, 'good'), (-1.78, 0.0, 'bad')])}</div>
      <div class="score"><div class="k">Altman Z″-Score</div>
        <div class="v">{z.z_double_prime:.2f}{_pill(z_zone + ' zone', z_tone)}</div>
        <div class="rule">Distress below 1.10 · safe above 2.60</div>
        {_scale(z.z_double_prime, 0.0, 6.0, [(0.0, 1.10, 'bad'), (1.10, 2.60, 'warn'), (2.60, 6.0, 'good')])}</div>
      <div class="score"><div class="k">Cash conversion cycle</div>
        <div class="v">{c.ccc_t:.1f} d{_pill(f'{c.delta_ccc:+.1f} d · {c_label}', c_tone)}</div>
        <div class="rule">Alert: +20 days with a stable gross margin</div>
        {_scale(c.delta_ccc, -30.0, 40.0, [(-30.0, 10.0, 'good'), (10.0, 20.0, 'warn'), (20.0, 40.0, 'bad')])}</div>
    </div>"""
    fi_note = (
        '<div class="note"><b>Financial-sector borrower.</b> Beneish, Altman Z″ and cash-cycle scores are calibrated '
        'on non-financial companies and were excluded from the risk band.</div>'
        if fx["is_fi"] else ""
    )
    beneish_idx = [("DSRI", b.dsri), ("GMI", b.gmi), ("AQI", b.aqi), ("SGI", b.sgi), ("DEPI", b.depi), ("SGAI", b.sgai), ("LVGI", b.lvgi), ("TATA", b.tata)]
    beneish_table = (
        "<table><tr>" + "".join(f'<th class="num">{name}</th>' for name, _ in beneish_idx) + "</tr><tr>"
        + "".join(f'<td class="num">{value:.3f}</td>' for _, value in beneish_idx) + "</tr></table>"
    )
    altman_rows = [
        ("Working capital ÷ assets", "X1", 6.56, z.x1),
        ("Retained earnings ÷ assets", "X2", 3.26, z.x2),
        ("EBIT ÷ assets", "X3", 6.72, z.x3),
        ("Book equity ÷ liabilities", "X4", 1.05, z.x4),
    ]
    altman_table = (
        '<table><tr><th>Ratio</th><th></th><th class="num">Value</th><th class="num">Weight</th><th class="num">Contribution</th></tr>'
        + "".join(f'<tr><td>{name}</td><td class="sym">{sym}</td><td class="num">{val:.3f}</td><td class="num">{w:.2f}</td>'
                  f'<td class="num">{w * val:.2f}</td></tr>' for name, sym, w, val in altman_rows)
        + f'<tr class="total"><td>Altman Z″-Score</td><td></td><td></td><td></td><td class="num">{z.z_double_prime:.2f}</td></tr></table>'
    )
    ccc_table = (
        '<table><tr><th>Days</th><th class="num">Prior year</th><th class="num">Latest year</th><th class="num">Change</th></tr>'
        + "".join(f'<tr><td>{name}</td><td class="num">{prev:.1f}</td><td class="num">{cur:.1f}</td><td class="num">{cur - prev:+.1f}</td></tr>'
                  for name, prev, cur in (("Receivables (DSO)", c.dso_t_minus_1, c.dso_t),
                                          ("Inventory (DIO)", c.dio_t_minus_1, c.dio_t),
                                          ("Payables (DPO)", c.dpo_t_minus_1, c.dpo_t)))
        + f'<tr class="total"><td>Cash cycle</td><td class="num">{c.ccc_t_minus_1:.1f}</td><td class="num">{c.ccc_t:.1f}</td>'
          f'<td class="num">{c.delta_ccc:+.1f}</td></tr></table>'
    )
    detail_tables = "" if fx["is_fi"] else f"""
<div class="block">
  <h3>Beneish indices</h3>
  {beneish_table}
</div>
<div class="kv" style="margin-top:6pt">
  <div class="block" style="flex:1.15"><h3>Altman Z″ components</h3>{altman_table}</div>
  <div class="block" style="flex:1"><h3>Working-capital cycle</h3>{ccc_table}</div>
</div>"""
    forensic_notes = "".join(f"<li>{_esc(n)}</li>" for n in fx["forensic_notes"])
    ecl_rows = "".join(
        "<tr>" + "".join(f"<td{NUM if i else ''}>{_esc(cell)}</td>" for i, cell in enumerate(row)) + "</tr>"
        for row in fx["ecl_rows"]
    )
    dd_note = ("Risk-neutral distance to default equals the physical one because the asset drift is set to the risk-free rate."
               if fx["dd_rn_same"] else "")

    running = fx["company"] + (f" ({fx['ticker']})" if fx["ticker"] else "")
    title = f"Credit Risk Memorandum — {running}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<meta name="author" content="Aegis Credit Risk Engine">
<meta name="description" content="{_esc(fx['band_label'])} · {_esc(fx['stage_name'])} · prepared {_esc(fx['prepared'])}">
<meta name="generator" content="Aegis Credit Risk Engine">
<style>{MEMO_CSS}</style>
</head>
<body>

<div class="masthead">
  <div>
    <div class="wordmark">AEGIS</div>
    <div class="doctype">Credit Risk Memorandum</div>
  </div>
  <div class="meta">
    <div>Prepared <b>{_esc(fx['prepared'])}</b></div>
    <div>Reference <b>{_esc(fx['reference'])}</b></div>
  </div>
</div>

<h1 class="borrower" data-run="{_esc(running)}">{_esc(fx['company'])}{f'<span class="ticker">{_esc(fx["ticker"])}</span>' if fx['ticker'] else ''}</h1>
<div class="basis">Data basis: {_esc(fx['basis'])}</div>

<div class="summary">
  <div class="verdict" style="background:{band_fg}">
    <div class="k">Internal risk band</div>
    <div class="band">{_esc(fx['band_label'])}</div>
    <div class="band-name">{_esc(fx['band_name'])}</div>
    <div class="stage">IFRS 9 · <b>{_esc(fx['stage_name'])}</b></div>
  </div>
  <div class="kpis">{kpis}</div>
</div>

<div class="stance"><b class="t">{_esc(fx['stance_title'])}</b>{(' — ' + _esc(fx['stance_text'])) if fx['stance_text'] else ''}</div>

<h3>Key findings</h3>
<ul class="findings">{findings}</ul>

<h2><span class="n">1</span>Inputs and sources</h2>
<div class="block">
<table>
  <colgroup><col style="width:23%"><col style="width:12%"><col style="width:17%"><col style="width:36%"><col style="width:12%"></colgroup>
  <tr><th>Input</th><th></th><th class="num">Value</th><th>Source</th><th class="num">As of</th></tr>
  {input_rows}
</table>
</div>

<h2><span class="n">2</span>Structural credit risk (Merton model)</h2>
{merton_table}
{f'<p class="method" style="margin-top:3pt">{_esc(dd_note)}</p>' if dd_note else ''}
<div class="figures">
  <div class="figure"><div class="cap">Capital structure</div>{_img(charts['capital_structure'], 'Capital structure')}</div>
  <div class="figure"><div class="cap">Distance to default <span>(default zone shaded)</span></div>{_img(charts['distance_to_default'], 'Distance to default')}</div>
</div>

<h2><span class="n">3</span>Financial statement forensics</h2>
<p class="method">Basis: {_esc(fx['forensic_basis'])} · composite forensic risk: <b>{_esc(fx['composite'])}</b></p>
{fi_note}
{scores}
{detail_tables}
{f'<h3>Forensic notes</h3><ul class="findings">{forensic_notes}</ul>' if forensic_notes else ''}

<h2><span class="n">4</span>Regulatory capital (Vasicek ASRF)</h2>
<div class="block">
  <div class="kv" style="align-items:flex-start">
    <div style="flex:1">{vasicek_table}</div>
    <div class="figure" style="flex:1.05"><div class="cap">Conditional PD under systematic stress</div>{_img(charts['vasicek_stress'], 'Vasicek stress curve')}</div>
  </div>
</div>

<h2><span class="n">5</span>IFRS 9 expected credit loss</h2>
<div class="block">
  <div class="stagebox">
    <div class="stagebadge" style="color:{stage_fg};background:{stage_bg}">{_esc(fx['stage_name'])}</div>
    <p>{_esc(fx['stage_rationale'])}</p>
  </div>
  <table>
    <tr><th>Period</th><th class="num">Marginal PD</th><th class="num">Cumulative PD</th><th class="num">Survival</th>
        <th class="num">EAD</th><th class="num">LGD</th><th class="num">Discount factor</th><th class="num">Discounted ECL</th></tr>
    {ecl_rows}
    <tr class="total"><td colspan="7">Total allowance · {_esc(fx['ecl_pct'])} of exposure at default</td><td class="num">{_esc(fx['ecl_total'])}</td></tr>
  </table>
</div>

<h2><span class="n">6</span>Method and limitations</h2>
<div class="method block">
  <ul>
    <li>Merton (1974) structural model: asset value and volatility are solved from the equity value and volatility; the default point is short-term debt plus half of long-term debt (KMV convention).</li>
    <li>Forensics: Beneish (1999) eight-index M-Score, Altman Z″ (1995) for non-manufacturing and emerging-market firms, and the cash conversion cycle; these models are not applied to financial companies.</li>
    <li>Regulatory capital: Vasicek single-factor model with the Basel IRB corporate correlation; the point-in-time PD is floored at 0.01% for capital and expected credit loss.</li>
    <li>IFRS 9 staging: Stage 2 when distance to default falls more than 1σ from origination, falls below {SICR_WATCHLIST_DD:.1f}σ, or payments are 30+ days past due; Stage 3 at 90+ days past due, a PD of 50% or more, or impairment.</li>
  </ul>
  <div class="disclaimer"><b>Disclaimer.</b> {_esc(fx['disclaimer'])} Model output for screening purposes only: not a credit rating, credit opinion or investment advice.</div>
</div>

</body>
</html>
"""


# --- PDF -----------------------------------------------------------------------------------------

def generate_credit_memo_pdf_reportlab(assessment: ComprehensiveCreditAssessment) -> bytes:
    """Fallback renderer (no Cairo/Pango needed): the same figures, plainer layout."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    fx = memo_facts(assessment)
    charts = generate_credit_charts(assessment, "png")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor(INK))
    small = ParagraphStyle("small", parent=body, fontSize=6.5, leading=9, textColor=colors.HexColor(MUTED))
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor(NAVY))
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=14, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor(NAVY))

    def table(rows, widths, header=True):
        t = Table(rows, colWidths=widths)
        style = [("FONT", (0, 0), (-1, -1), "Helvetica", 7.5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(HAIRLINE)),
                 ("ALIGN", (2, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if header:
            style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.8), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(NAVY))]
        t.setStyle(TableStyle(style))
        return t

    width = 180 * mm
    story = [
        Paragraph("AEGIS · CREDIT RISK MEMORANDUM", small),
        Paragraph(_esc(fx["company"]) + (f" ({_esc(fx['ticker'])})" if fx["ticker"] else ""), h1),
        Paragraph(f"Prepared {_esc(fx['prepared'])} · Reference {_esc(fx['reference'])} · {_esc(fx['basis'])}", small),
        Spacer(1, 6),
        table([["Risk band", "IFRS 9 stage"] + [k for k, _, _ in fx["kpis"][:3]],
               [f"{fx['band_label']} ({fx['band_name']})", fx["stage_name"]] + [v for _, v, _ in fx["kpis"][:3]]],
              [width * w for w in (0.22, 0.26, 0.17, 0.18, 0.17)]),
        Spacer(1, 4),
        Paragraph(f"<b>{_esc(fx['stance_title'])}</b> — {_esc(fx['stance_text'])}", body),
        Spacer(1, 4),
    ] + [Paragraph("• " + _esc(item), body) for item in fx["findings"]] + [
        Paragraph("1  Inputs and sources", h2),
        table([["Input", "", "Value", "Source", "As of"]] + [list(r) for r in fx["input_rows"]],
              [width * w for w in (0.24, 0.13, 0.15, 0.36, 0.12)]),
        Paragraph("2  Structural credit risk (Merton model)", h2),
        table([[n, s, v] for n, s, v in fx["merton_rows"]], [width * 0.55, width * 0.15, width * 0.30], header=False),
        Spacer(1, 4),
        Table([[Image(io.BytesIO(charts["capital_structure"]), width=width * 0.48, height=width * 0.48 * 1.75 / 3.4),
                Image(io.BytesIO(charts["distance_to_default"]), width=width * 0.48, height=width * 0.48 * 1.75 / 3.4)]]),
        Paragraph("3  Financial statement forensics", h2),
        Paragraph(f"Basis: {_esc(fx['forensic_basis'])} · composite forensic risk: {_esc(fx['composite'])}", small),
    ]
    if fx["is_fi"]:
        story.append(Paragraph("Financial-sector borrower: Beneish, Altman Z'' and cash-cycle scores are calibrated on "
                               "non-financial companies and were excluded from the risk band.", body))
    b, z, c = fx["beneish"], fx["altman"], fx["ccc"]
    scores = (["n/a"] * 5 if fx["is_fi"] else
              [f"{b.m_score:.2f}", f"{z.z_double_prime:.2f}", (z.zone or "").title(), f"{c.ccc_t:.1f} days", f"{c.delta_ccc:+.1f} days"])
    story += [
        table([["Beneish M-Score", "Altman Z''", "Zone", "Cash cycle", "Change"], scores], [width / 5] * 5),
        Paragraph("4  Regulatory capital (Vasicek ASRF)", h2),
        table([[k, v] for k, v in fx["vasicek_rows"]], [width * 0.6, width * 0.4], header=False),
        Image(io.BytesIO(charts["vasicek_stress"]), width=width * 0.55, height=width * 0.55 * 1.75 / 3.4),
        Paragraph("5  IFRS 9 expected credit loss", h2),
        Paragraph(f"<b>{_esc(fx['stage_name'])}</b>: {_esc(fx['stage_rationale'])}", body),
        table([["Period", "Marginal PD", "Cumulative PD", "Survival", "EAD", "LGD", "Discount", "Discounted ECL"]]
              + [list(r) for r in fx["ecl_rows"]] + [["Total", "", "", "", "", "", fx["ecl_pct"], fx["ecl_total"]]],
              [width / 8] * 8),
        Spacer(1, 8),
        Paragraph(f"<b>Disclaimer.</b> {_esc(fx['disclaimer'])} Model output for screening purposes only: not a credit "
                  "rating, credit opinion or investment advice.", small),
    ]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                      title=f"Credit Risk Memorandum — {fx['company']}", author="Aegis Credit Risk Engine").build(story)
    return buffer.getvalue()


def generate_credit_memo_pdf(assessment: ComprehensiveCreditAssessment) -> bytes:
    """Compiles the memo to PDF with WeasyPrint, or with ReportLab where WeasyPrint can't run."""
    try:
        import weasyprint
    except (ImportError, OSError):            # missing package or missing Cairo/Pango libraries
        return generate_credit_memo_pdf_reportlab(assessment)
    return weasyprint.HTML(string=generate_credit_memo_html(assessment)).write_pdf()


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
