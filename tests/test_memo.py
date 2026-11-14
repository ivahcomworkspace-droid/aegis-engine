"""
The credit memo prints exactly what the dashboard shows: both come from one engine assessment and
use the same number formats. These tests pin that, plus the inputs the memo depends on.
"""

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.vasicek import calculate_ifrs9_ecl
from schemas.credit_models import IFRS9StageInput
from services.formatting import format_money_m, format_pd
from services.memo_generator import generate_credit_charts, memo_facts
from schemas.credit_models import ComprehensiveCreditResponse

client = TestClient(app)
DASHBOARD = (Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")


def _js_object(name: str) -> dict:
    """A plain object literal from the dashboard script (unquoted keys, numeric values)."""
    body = re.search(rf"const {name} = \{{(.*?)\}};", DASHBOARD, re.S).group(1)
    return json.loads("{" + re.sub(r"(\w+):", r'"\1":', body) + "}")


def _slider_default(slider_id: str) -> float:
    return float(re.search(rf'id="{slider_id}"[^>]*value="([\d.]+)"', DASHBOARD).group(1))


def _sample_payload(**changes) -> dict:
    """What the dashboard sends for the sample company with its default sliders."""
    E, vol, std, ltd, r = (_slider_default(i) for i in ("rngEquity", "rngEqVol", "rngSTD", "rngLTD", "rngRFR"))
    payload = {
        "company_name": "Apex Industrial Corp", "ticker": "APEX",
        "equity_market_cap": E, "equity_volatility": vol / 100, "short_term_debt": std, "long_term_debt": ltd,
        "risk_free_rate": r / 100,
        "merton_input": {"equity_value": E, "equity_volatility": vol / 100, "short_term_debt": std,
                         "long_term_debt": ltd, "risk_free_rate": r / 100, "maturity": 1.0, "loss_given_default": 0.45},
        "forensic_input": _js_object("SAMPLE_FORENSIC_INPUT"),
        "is_financial_institution": False,
        "loan_exposure": max(std + ltd, 100.0),
        "loan_tenor_years": 5,
        "input_sources": {key: {"source": "Sample scenario"} for key in
                          ("equity_value", "equity_volatility", "short_term_debt", "long_term_debt", "risk_free_rate")},
        "forensic_basis": "Sample financial statements (illustrative)",
    }
    payload.update(changes)
    return payload


# --- One assessment, one set of formats ------------------------------------------------------------

def _js_function(name: str) -> str:
    start = DASHBOARD.index(f"function {name}(")
    depth, i = 0, DASHBOARD.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(DASHBOARD[i], 0)
        if depth == 0:
            return DASHBOARD[start:i + 1]
        i += 1


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the dashboard's formatters")
def test_number_formats_match_the_dashboard_code():
    money = [0, 0.000018, 0.0055, 0.5, 1.0, 12.34, 276.4798, 999.9, 1000, 4530.2, 12345, 183900, 5177380.48, 2.5e6]
    pds = [0, 1e-82, 3.7254598960523034e-07, 4.8e-7, 2e-6, 5e-5, 0.0001, 0.00102, 0.0231, 0.5]
    script = (_js_function("formatMoneyM") + "\n" + _js_function("formatPD") + "\n"
              + f"console.log(JSON.stringify([{json.dumps(money)}.map(formatMoneyM), {json.dumps(pds)}.map(formatPD)]));")
    js_money, js_pd = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert [format_money_m(v) for v in money] == js_money
    assert [format_pd(p) for p in pds] == js_pd


def test_memo_prints_the_engines_figures():
    payload = _sample_payload()
    a = client.post("/api/v1/credit/comprehensive", json=payload).json()
    memo = client.post("/api/v1/credit/memo/html", json=payload).text
    m, ecl, f = a["merton"], a["ifrs9_ecl"], a["forensics"]
    band, _, band_name = a["internal_risk_band"].partition(" (")
    stance_title, _, stance_text = a["indicative_stance"].partition(" — ")
    expected = [
        f"{m['distance_to_default_physical']:.2f}σ",
        format_pd(m["default_probability_physical"]),
        format_money_m(m["asset_value"]),
        f"{m['credit_spread_bps']:.1f} bps",
        format_money_m(ecl["total_ecl"]),
        f"{ecl['ecl_percentage_of_ead']:.3f}%",
        band, band_name.rstrip(")"), stance_title, stance_text, ecl["stage_name"],
        f"{f['beneish']['m_score']:.2f}", f"{f['altman']['z_double_prime']:.2f}", f"{f['ccc']['ccc_t']:.1f} d",
        format_money_m(payload["equity_market_cap"]),          # the market cap the user set, not V - D
    ]
    for value in expected:
        assert html.escape(value) in memo, value
    assert "PRIME / STABLE QUALITY" not in memo                # the memo no longer overrides the engine


def test_memo_lists_inputs_with_their_sources():
    payload = _sample_payload(input_sources={
        "equity_value": {"source": "Yahoo Finance (unofficial) price × SEC shares", "as_of": "2026-11-13"},
        "short_term_debt": {"source": "SEC 10-Q balance sheet", "as_of": "2026-09-30"},
    }, forensic_basis="Filed annual reports (SEC 10-K), FY2026 vs FY2025")
    a = client.post("/api/v1/credit/comprehensive", json=payload).json()
    assert a["inputs"]["equity_value"] == payload["equity_market_cap"]
    assert a["inputs"]["equity_volatility"] == payload["equity_volatility"]
    assert a["input_sources"]["short_term_debt"] == {"source": "SEC 10-Q balance sheet", "as_of": "2026-09-30"}
    facts = memo_facts(ComprehensiveCreditResponse(**a))
    rows = {name: (value, source, as_of) for name, _sym, value, source, as_of in facts["input_rows"]}
    assert rows["Market value of equity"] == ("$200.0M", "Yahoo Finance (unofficial) price × SEC shares", "2026-11-13")
    assert rows["Equity volatility"][1] == "Supplied with the request"
    assert facts["basis"].startswith("Live public data")
    assert facts["forensic_basis"] == "Filed annual reports (SEC 10-K), FY2026 vs FY2025"


def test_small_allowances_keep_their_value():
    """Money is in $ millions: rounding to 2 decimals turned a $5.5K allowance into $10K or $0."""
    a = client.post("/api/v1/credit/comprehensive", json=_sample_payload()).json()
    assert format_money_m(a["ifrs9_ecl"]["total_ecl"]) == "$5.5k"
    assert "ECL allowance of $5.5k (0.004% of EAD)" in a["executive_summary"]
    assert format_money_m(a["vasicek"]["expected_loss"]) == "$5.9k"


def test_distance_to_default_below_two_sigma_is_stage_2():
    res = calculate_ifrs9_ecl(IFRS9StageInput(
        current_distance_to_default=1.5, origination_distance_to_default=1.8, current_pd=0.067,
        days_past_due=0, ead_schedule=[100.0, 80.0, 60.0],
    ))
    assert res.assigned_stage == 2
    assert "watchlist" in res.stage_rationale
    assert len(res.period_schedules) == 3


def test_bank_memo_shows_no_placeholder_scores():
    from tests.test_market_data import _bank_like_request
    memo = client.post("/api/v1/credit/memo/html", json=_bank_like_request(True)).text
    assert "excluded from the risk band" in memo
    assert memo.count(">n/a<") == 3
    assert "Altman Z″ components" not in memo


def test_charts_are_vector():
    a = ComprehensiveCreditResponse(**client.post("/api/v1/credit/comprehensive", json=_sample_payload()).json())
    charts = generate_credit_charts(a)
    assert set(charts) == {"capital_structure", "distance_to_default", "vasicek_stress"}
    assert all(b"<svg" in svg for svg in charts.values())


# --- The sample company ---------------------------------------------------------------------------

def test_sample_statements_describe_one_consistent_company():
    """The sample's statements must balance and carry the debt the sample sliders show."""
    s = _js_object("SAMPLE_FORENSIC_INPUT")
    assert s["total_assets_t"] == pytest.approx(s["total_liabilities_t"] + s["book_value_equity_t"])
    for t in ("_t", "_t_minus_1"):
        assert s[f"current_assets{t}"] + s[f"ppe{t}"] <= s[f"total_assets{t}"]
    assert s["current_liabilities_t"] + s["long_term_debt_t"] <= s["total_liabilities_t"]
    assert s["long_term_debt_t_minus_1"] <= s["total_liabilities_t_minus_1"]
    assert s["receivables_t"] + s["inventory_t"] <= s["current_assets_t"]
    assert s["long_term_debt_t"] == _slider_default("rngLTD")
    assert _slider_default("rngSTD") <= s["current_liabilities_t"]
