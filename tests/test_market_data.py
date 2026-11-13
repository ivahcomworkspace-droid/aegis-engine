"""
Offline tests for live borrower data (services/market_data.py and /api/v1/market/borrower).
Network calls are replaced with synthetic SEC, Treasury and price data.
"""

import math

import pytest
from fastapi.testclient import TestClient

from api.main import app
from schemas.credit_models import ForensicInput
from services import market_data as md

client = TestClient(app)

SHEET = "2026-09-30"          # latest quarter
FY = "2026-06-30"             # latest fiscal year end
FY_PRIOR = "2025-06-30"


def row(val, end, form="10-Q", filed=None, start=None, accn="0000000001-26-000001"):
    fact = {"val": val, "end": end, "form": form, "filed": filed or end, "accn": accn}
    if start:
        fact["start"] = start
    return fact


def facts_json(gaap, dei=None, name="TEST CORP"):
    def concept(rows, unit="USD"):
        return {"units": {unit: rows}}

    payload = {"entityName": name, "facts": {"us-gaap": {}, "dei": {}}}
    for key, rows in gaap.items():
        unit = "shares" if key.endswith("SharesOutstanding") else "USD"
        payload["facts"]["us-gaap"][key] = concept(rows, unit)
    for key, rows in (dei or {}).items():
        payload["facts"]["dei"][key] = concept(rows, "shares")
    return payload


def annual(val, end, start):
    return row(val, end, form="10-K", filed="2026-08-01", start=start)


def industrial_gaap():
    """A non-financial filer with a quarterly balance sheet and two annual reports ($)."""
    m = 1_000_000
    fy, fy0 = (FY, "2025-07-01"), (FY_PRIOR, "2024-07-01")
    return {
        "Assets": [row(780_000 * m, SHEET), row(760_000 * m, FY, "10-K"), row(620_000 * m, FY_PRIOR, "10-K")],
        "LongTermDebtCurrent": [row(4_391 * m, SHEET), row(2_000 * m, FY, "10-K"), row(2_500 * m, FY_PRIOR, "10-K")],
        "CommercialPaper": [row(2_000 * m, SHEET)],
        "ShortTermBorrowings": [row(2_000 * m, SHEET), row(9_999 * m, "2018-06-30", "10-K")],
        "LongTermDebtNoncurrent": [row(31_090 * m, SHEET), row(31_000 * m, FY, "10-K"), row(40_000 * m, FY_PRIOR, "10-K")],
        "Liabilities": [row(310_000 * m, SHEET), row(316_000 * m, FY, "10-K"), row(276_000 * m, FY_PRIOR, "10-K")],
        "StockholdersEquity": [row(470_000 * m, SHEET), row(444_000 * m, FY, "10-K"), row(344_000 * m, FY_PRIOR, "10-K")],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [annual(330_000 * m, *fy), annual(280_000 * m, *fy0)],
        "CostOfRevenue": [annual(106_000 * m, *fy), annual(90_000 * m, *fy0)],
        "SellingAndMarketingExpense": [annual(25_000 * m, *fy), annual(24_000 * m, *fy0)],
        "GeneralAndAdministrativeExpense": [annual(9_000 * m, *fy), annual(8_000 * m, *fy0)],
        "DepreciationDepletionAndAmortization": [annual(34_000 * m, *fy), annual(30_000 * m, *fy0)],
        "NetIncomeLoss": [annual(130_000 * m, *fy), annual(100_000 * m, *fy0)],
        "NetCashProvidedByUsedInOperatingActivities": [annual(180_000 * m, *fy), annual(130_000 * m, *fy0)],
        "OperatingIncomeLoss": [annual(155_000 * m, *fy), annual(128_000 * m, *fy0)],
        "AccountsReceivableNetCurrent": [row(80_000 * m, FY, "10-K"), row(69_000 * m, FY_PRIOR, "10-K")],
        "AssetsCurrent": [row(200_000 * m, FY, "10-K"), row(190_000 * m, FY_PRIOR, "10-K")],
        "LiabilitiesCurrent": [row(150_000 * m, FY, "10-K"), row(140_000 * m, FY_PRIOR, "10-K")],
        "PropertyPlantAndEquipmentNet": [row(300_000 * m, FY, "10-K"), row(200_000 * m, FY_PRIOR, "10-K")],
        "InventoryNet": [row(1_400 * m, FY, "10-K"), row(1_200 * m, FY_PRIOR, "10-K")],
        "AccountsPayableCurrent": [row(42_000 * m, FY, "10-K"), row(27_000 * m, FY_PRIOR, "10-K")],
        "RetainedEarningsAccumulatedDeficit": [row(330_000 * m, FY, "10-K"), row(250_000 * m, FY_PRIOR, "10-K")],
        "MarketableSecuritiesCurrent": [row(60_000 * m, FY, "10-K"), row(55_000 * m, FY_PRIOR, "10-K")],
    }


def industrial_facts():
    dei = {"EntityCommonStockSharesOutstanding": [row(7_426_194_600, "2026-10-22")]}
    return facts_json(industrial_gaap(), dei)


def bank_facts():
    m = 1_000_000
    gaap = {
        "Assets": [row(5_091_545 * m, SHEET)],
        "Liabilities": [row(4_719_177 * m, SHEET)],
        "StockholdersEquity": [row(372_368 * m, SHEET)],
        "Deposits": [row(2_780_793 * m, SHEET)],
        "FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase": [row(703_392 * m, SHEET)],
        "ShortTermBorrowings": [row(71_339 * m, SHEET)],
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": [row(460_160 * m, SHEET)],
    }
    return facts_json(gaap, {"EntityCommonStockSharesOutstanding": [row(2_655_609_500, SHEET)]}, "BIG BANK")


def price_history(price=533.16, n=253, daily=0.01):
    closes = [100.0 * math.exp(daily * (1 if i % 2 else -1) * (i % 2)) for i in range(n)]
    return md.PriceHistory(
        symbol="TEST", currency="USD", last_price=price, last_date="2026-11-10",
        closes=closes, display_name="Test Corporation", source="Test feed",
    )


# --- Debt, shares and bank rules ---------------------------------------------------------------

def test_industrial_debt_uses_current_portion_plus_largest_short_term_borrowing():
    debt = md.extract_debt(md.CompanyFacts(industrial_facts()), SHEET, bank=False)
    # Commercial paper and short-term borrowings overlap, so only the larger counts; the 2018 value is stale
    assert debt.short_term_m == pytest.approx(4_391 + 2_000)
    assert debt.long_term_m == pytest.approx(31_090)
    assert {c.term for c in debt.components} == {"short", "long"}


def test_debt_current_total_takes_precedence_over_components():
    gaap = industrial_gaap()
    gaap["DebtCurrent"] = [row(7_000 * 1e6, SHEET)]
    debt = md.extract_debt(md.CompanyFacts(facts_json(gaap)), SHEET, bank=False)
    assert debt.short_term_m == pytest.approx(7_000)


def test_long_term_debt_derived_from_total_less_current_portion():
    gaap = industrial_gaap()
    del gaap["LongTermDebtNoncurrent"]
    gaap["LongTermDebt"] = [row(35_481 * 1e6, SHEET)]
    debt = md.extract_debt(md.CompanyFacts(facts_json(gaap)), SHEET, bank=False)
    assert debt.long_term_m == pytest.approx(35_481 - 4_391)


def test_bank_counts_deposits_and_repo_funding_as_short_term_debt():
    facts = md.CompanyFacts(bank_facts())
    assert md.is_deposit_taker("6021", facts, SHEET)
    debt = md.extract_debt(facts, SHEET, bank=True)
    assert debt.short_term_m == pytest.approx(2_780_793 + 703_392 + 71_339)
    assert debt.long_term_m == pytest.approx(460_160)
    assert any("deposits" in note for note in debt.notes)


def test_deposit_heavy_filer_is_treated_as_bank_whatever_its_sic_code():
    facts = md.CompanyFacts(bank_facts())
    assert md.is_deposit_taker("6211", facts, SHEET)
    assert not md.is_deposit_taker("7372", md.CompanyFacts(industrial_facts()), SHEET)


def test_untagged_debt_falls_back_to_current_and_other_liabilities():
    """Ford and GM keep debt under their own tags; liabilities stand in (KMV default point)."""
    gaap = industrial_gaap()
    for concept in ("LongTermDebtCurrent", "CommercialPaper", "ShortTermBorrowings", "LongTermDebtNoncurrent"):
        del gaap[concept]
    gaap["LiabilitiesCurrent"].append(row(111_967 * 1e6, SHEET))
    debt = md.extract_debt(md.CompanyFacts(facts_json(gaap)), SHEET, bank=False)
    assert debt.short_term_m == pytest.approx(111_967)
    assert debt.long_term_m == pytest.approx(310_000 - 111_967)
    assert "liabilities stand in" in debt.notes[0]


def test_unclassified_balance_sheet_counts_all_liabilities_as_long_term():
    """Berkshire's balance sheet has no current/long-term split."""
    gaap = {k: v for k, v in industrial_gaap().items()
            if k in ("Assets", "Liabilities", "StockholdersEquity")}
    debt = md.extract_debt(md.CompanyFacts(facts_json(gaap)), SHEET, bank=False)
    assert debt.short_term_m == 0
    assert debt.long_term_m == pytest.approx(310_000)
    assert "all liabilities count as long-term debt" in debt.notes[0]


COVER_XML = """<xbrl>
  <context id="c-1"><entity><segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassAMember</xbrldi:explicitMember></segment></entity><period><instant>2026-10-28</instant></period></context>
  <context id="c-2"><entity><segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassBMember</xbrldi:explicitMember></segment></entity><period><instant>2026-10-28</instant></period></context>
  <context id="c-3"><entity><segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassCMember</xbrldi:explicitMember></segment></entity><period><instant>2026-10-28</instant></period></context>
  <context id="c-4"><entity><segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">x:SeniorNotesDue2030Member</xbrldi:explicitMember></segment></entity><period><startDate>2026-07-01</startDate><endDate>2026-09-30</endDate></period></context>
  <dei:TradingSymbol contextRef="c-1" id="f1">BRK.A</dei:TradingSymbol>
  <dei:TradingSymbol contextRef="c-2" id="f2">BRK.B</dei:TradingSymbol>
  <dei:TradingSymbol contextRef="c-4" id="f3">BRK30</dei:TradingSymbol>
  <dei:EntityCommonStockSharesOutstanding contextRef="c-1" decimals="INF" unitRef="shares" id="f4">480476</dei:EntityCommonStockSharesOutstanding>
  <dei:EntityCommonStockSharesOutstanding contextRef="c-2" decimals="INF" unitRef="shares" id="f5">1403805594</dei:EntityCommonStockSharesOutstanding>
  <dei:EntityCommonStockSharesOutstanding contextRef="c-3" decimals="INF" unitRef="shares" id="f6">9080220</dei:EntityCommonStockSharesOutstanding>
</xbrl>"""


def test_cover_page_gives_shares_and_ticker_per_class():
    classes = md.parse_cover_share_classes(COVER_XML)
    assert classes == [
        {"label": "Class A", "shares_m": 0.480476, "symbol": "BRK-A", "as_of": "2026-10-28"},
        {"label": "Class B", "shares_m": 1403.805594, "symbol": "BRK-B", "as_of": "2026-10-28"},
        {"label": "Class C", "shares_m": 9.08022, "symbol": None, "as_of": "2026-10-28"},
    ]


def test_shares_fall_back_to_balance_sheet_count_for_multi_class_filers():
    gaap = industrial_gaap()
    gaap["CommonStockSharesOutstanding"] = [row(12_294_000_000, SHEET)]
    shares = md.extract_shares(md.CompanyFacts(facts_json(gaap)), SHEET)
    assert shares == (12_294.0, SHEET, "balance sheet")


def test_snapshot_rejects_non_us_gaap_filers_with_guidance():
    with pytest.raises(md.MarketDataError) as err:
        md.build_sec_snapshot({"entityName": "IFRS CO", "facts": {"ifrs-full": {"Assets": {}}}}, {})
    assert err.value.status_code == 422
    assert "by hand" in err.value.message


# --- Annual financials for the forensic models ---------------------------------------------------

def test_financials_build_a_valid_forensic_input():
    financials, notes = md.extract_financials(md.CompanyFacts(industrial_facts()), bank=False)
    assert notes == []
    assert financials.fiscal_year_end == FY and financials.prior_fiscal_year_end == FY_PRIOR
    data = financials.forensic_input
    ForensicInput(**data)
    assert data["sales_t"] == pytest.approx(330_000)
    assert data["sga_t"] == pytest.approx(34_000)          # selling & marketing + G&A
    assert data["long_term_debt_t_minus_1"] == pytest.approx(40_000)
    assert data["working_capital_t"] == pytest.approx(50_000)


def test_untagged_optional_lines_are_counted_as_zero_and_disclosed():
    """AGENT_RULES.md rule 1: no silent defaults for missing accounting lines."""
    gaap = industrial_gaap()
    del gaap["InventoryNet"]
    financials, _ = md.extract_financials(md.CompanyFacts(facts_json(gaap)), bank=False)
    assert financials.forensic_input["inventory_t"] == 0
    assert any("counted as 0: inventory (2026), inventory (2025)" in n for n in financials.notes)


def test_financials_missing_revenue_keep_sample_with_a_note():
    gaap = industrial_gaap()
    del gaap["RevenueFromContractWithCustomerExcludingAssessedTax"]
    financials, notes = md.extract_financials(md.CompanyFacts(facts_json(gaap)), bank=False)
    assert financials is None
    assert "revenue" in notes[0]


def test_financials_reconcile_temporary_equity_into_liabilities():
    gaap = industrial_gaap()
    gaap["Liabilities"] = [row(250_000 * 1e6, SHEET), row(250_000 * 1e6, FY, "10-K"), row(276_000 * 1e6, FY_PRIOR, "10-K")]
    financials, _ = md.extract_financials(md.CompanyFacts(facts_json(gaap)), bank=False)
    data = financials.forensic_input
    assert data["total_liabilities_t"] == pytest.approx(760_000 - 444_000)
    assert any("Temporary equity" in n for n in financials.notes)


# --- Treasury, prices and tickers -----------------------------------------------------------------

def test_treasury_csv_uses_latest_one_year_yield():
    csv_text = 'Date,"1 Mo","1 Yr","2 Yr"\n11/10/2026,4.05,4.56,4.73\n11/09/2026,4.04,4.38,4.56\n11/06/2026,4.00,,4.50\n'
    assert md.parse_treasury_csv(csv_text) == ("2026-11-10", pytest.approx(0.0456))
    assert md.par_yield_to_continuous(0.0456) == pytest.approx(2 * math.log(1.0228))


def test_volatility_is_annualised_daily_log_return_stdev():
    closes = [100.0, 101.0] * 130
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:])][-252:]
    mean = sum(returns) / len(returns)
    expected = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1) * 252)
    assert md.annualized_volatility(closes) == pytest.approx(expected)
    with pytest.raises(ValueError):
        md.annualized_volatility([100.0, 101.0, 102.0])


def test_yahoo_chart_prefers_adjusted_closes():
    payload = {"chart": {"result": [{
        "meta": {"symbol": "MSFT", "currency": "USD", "regularMarketPrice": 533.16,
                 "regularMarketTime": 1794344401, "longName": "Microsoft Corporation"},
        "timestamp": [1794258000, 1794344401],
        "indicators": {"quote": [{"close": [500.0, 533.16]}], "adjclose": [{"adjclose": [499.0, 533.16]}]},
    }], "error": None}}
    history = md.parse_yahoo_chart(payload, "Yahoo Finance (unofficial)")
    assert history.closes == [499.0, 533.16]
    assert history.display_name == "Microsoft Corporation"
    assert history.last_date == "2026-11-10"


def test_ticker_normalisation():
    assert md.normalize_ticker(" brk.b ") == "BRK-B"
    with pytest.raises(md.MarketDataError):
        md.normalize_ticker("not a ticker!")


def test_sec_contact_is_required(monkeypatch):
    monkeypatch.delenv("AEGIS_SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(md, "_dotenv_values", lambda: {})
    with pytest.raises(md.MarketDataError) as err:
        md.sec_user_agent()
    assert err.value.status_code == 503
    assert "AEGIS_SEC_USER_AGENT" in err.value.message


# --- API endpoint -------------------------------------------------------------------------------

@pytest.fixture
def offline_sources(monkeypatch):
    """Replaces the network calls behind /api/v1/market/borrower."""
    monkeypatch.setenv("AEGIS_SEC_USER_AGENT", "Aegis Tests tests@example.invalid")
    state = {"facts": industrial_facts(), "submissions": {"name": "TEST CORP", "sic": "7372"},
             "prices": price_history(), "rate": ("2026-11-10", 0.0456)}

    async def resolve(_client, ticker):
        if ticker == "NOPE":
            raise md.MarketDataError(404, "NOPE isn't among the companies that file with the SEC.")
        return "0000000001", "TEST CORP", state.get("symbols", [ticker])

    async def snapshot(_client, _cik):
        snap = md.build_sec_snapshot(state["facts"], state["submissions"])
        snap["share_classes"] = state.get("share_classes", [])
        return snap

    async def prices(_client, symbol):
        if isinstance(state["prices"], Exception):
            raise state["prices"]
        return state.get("prices_by_symbol", {}).get(symbol, state["prices"])

    async def rate(_client):
        return state["rate"]

    monkeypatch.setattr(md, "resolve_ticker", resolve)
    monkeypatch.setattr(md, "sec_snapshot", snapshot)
    monkeypatch.setattr(md, "price_history", prices)
    monkeypatch.setattr(md, "treasury_one_year", rate)
    return state


def test_api_borrower_fills_all_five_inputs(offline_sources):
    res = client.get("/api/v1/market/borrower", params={"ticker": "test"})
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "TEST"
    assert data["company_name"] == "Test Corporation"
    assert data["equity_value"]["value"] == pytest.approx(533.16 * 7_426.1946, rel=1e-6)
    assert 0 < data["equity_volatility"]["value"] < 1
    assert data["short_term_debt"]["value"] == pytest.approx(6_391)
    assert data["long_term_debt"]["value"] == pytest.approx(31_090)
    assert data["risk_free_rate"]["value"] == pytest.approx(2 * math.log(1.0228), rel=1e-4)
    assert data["is_bank"] is False and data["forensics_applicable"] is True
    assert data["forensic_analysis"]["altman"]["zone"]
    assert data["warnings"] == []


def test_api_borrower_still_loads_debt_and_rate_when_prices_fail(offline_sources):
    offline_sources["prices"] = RuntimeError("HTTP 429")
    data = client.get("/api/v1/market/borrower", params={"ticker": "TEST"}).json()
    assert data["equity_value"]["value"] is None
    assert data["equity_volatility"]["value"] is None
    assert data["short_term_debt"]["value"] == pytest.approx(6_391)
    assert any("Market prices are unavailable" in w for w in data["warnings"])


def test_api_borrower_marks_banks(offline_sources):
    offline_sources["facts"] = bank_facts()
    offline_sources["submissions"] = {"name": "BIG BANK", "sic": "6021", "sicDescription": "National Commercial Banks"}
    data = client.get("/api/v1/market/borrower", params={"ticker": "TEST"}).json()
    assert data["is_bank"] is True
    assert data["forensics_applicable"] is False
    assert data["short_term_debt"]["value"] == pytest.approx(3_555_524)


def test_api_borrower_prices_each_listed_share_class(offline_sources):
    """Berkshire-style: no total share count, two listed classes priced separately."""
    gaap = industrial_gaap()
    offline_sources["facts"] = facts_json(gaap)             # no shares in company facts
    offline_sources["share_classes"] = md.parse_cover_share_classes(COVER_XML)
    offline_sources["prices"] = price_history(price=470.0)   # BRK-B, the typed ticker
    offline_sources["prices_by_symbol"] = {"BRK-A": price_history(price=705_000.0)}
    data = client.get("/api/v1/market/borrower", params={"ticker": "BRK.B"}).json()
    assert data["ticker"] == "BRK-B"
    expected = 0.480476 * 705_000.0 + 1403.805594 * 470.0
    assert data["equity_value"]["value"] == pytest.approx(expected)
    assert "BRK-A" in data["equity_value"]["detail"] and "BRK-B" in data["equity_value"]["detail"]
    assert any("Class C shares aren't listed" in w for w in data["warnings"])


def test_predecessor_annual_reports_fill_the_memo(monkeypatch, tmp_path):
    """A new holding company (no 10-K yet) takes its annual reports from the registrant it replaced."""
    monkeypatch.setenv("AEGIS_SEC_USER_AGENT", "Aegis Tests tests@example.invalid")
    monkeypatch.setattr(md, "CACHE", md.JsonCache(tmp_path))
    monkeypatch.setitem(md.PREDECESSOR_CIKS, "0000000002", ("0000000001", "Old Parent Corp"))
    successor = {k: [r for r in v if r["form"] == "10-Q"] for k, v in industrial_gaap().items()}
    successor = facts_json({k: v for k, v in successor.items() if v},
                           {"EntityCommonStockSharesOutstanding": [row(7_426_194_600, "2026-10-22")]}, "NEW PARENT")
    responses = {
        md.SEC_FACTS_URL.format(cik="0000000002"): successor,
        md.SEC_SUBMISSIONS_URL.format(cik="0000000002"): {"name": "NEW PARENT", "sic": "2911"},
        md.SEC_FACTS_URL.format(cik="0000000001"): industrial_facts(),
    }

    async def fake_sec_json(_client, url, timeout):
        return responses.get(url)

    monkeypatch.setattr(md, "_sec_json", fake_sec_json)
    import asyncio
    snap = asyncio.run(md.sec_snapshot(None, "0000000002"))
    assert snap["financials"]["fiscal_year_end"] == FY
    assert snap["financial_notes"] == []
    assert any("Old Parent Corp" in n for n in snap["financials"]["notes"])


def test_api_borrower_insurer_gets_no_forensic_scores(offline_sources):
    """Insurers (SIC 6331) aren't deposit takers but are financial companies all the same."""
    offline_sources["submissions"] = {"name": "INSURER CO", "sic": "6331", "sicDescription": "Fire, Marine & Casualty Insurance"}
    data = client.get("/api/v1/market/borrower", params={"ticker": "TEST"}).json()
    assert data["is_bank"] is False
    assert data["forensics_applicable"] is False
    assert any("financial company's risk band" in w for w in data["warnings"])


def test_cash_cycle_without_cost_of_sales_uses_receivable_days_only():
    """Service companies report no cost of sales; inventory and payable days are then left out."""
    financials, _ = md.extract_financials(md.CompanyFacts(industrial_facts()), bank=False)
    data = dict(financials.forensic_input, cogs_t=0.0, cogs_t_minus_1=0.0)
    from core.forensics import calculate_ccc
    ccc = calculate_ccc(ForensicInput(**data))
    assert ccc.dio_t == 0 and ccc.dpo_t == 0
    assert ccc.ccc_t == pytest.approx(80_000 / 330_000 * 365, abs=0.01)
    assert abs(ccc.delta_ccc) < 365


def test_common_tickers_leave_out_preferreds_warrants_units_and_notes():
    assert md.common_tickers(["GOOGL", "GOOG", "GOOGM", "GOOGN"]) == ["GOOGL", "GOOG"]
    assert md.common_tickers(["PCG", "PCG-PX"]) == ["PCG"]
    assert md.common_tickers(["HEI", "HEI-A"]) == ["HEI", "HEI-A"]
    assert md.common_tickers(["RILY", "RILYG", "RILYL", "RILYP"]) == ["RILY"]
    assert md.common_tickers(["TDSPRV", "TDS", "TDS-PU"]) == ["TDS"]
    assert md.common_tickers(["DMAA", "DMAAR", "DMAAU"]) == ["DMAA"]
    assert md.common_tickers(["ETSS", "ETSS-UN", "ETSS-WT"]) == ["ETSS"]
    assert md.common_tickers(["NWSA", "NWS", "NWSLL"]) == ["NWSA", "NWS"]
    assert md.common_tickers(["SCE-PG", "SCE-PL"]) == []
    assert md.common_tickers(["BMY", "CELG-RI"]) == ["BMY"]


def test_directory_drops_the_mark_when_a_company_has_no_common_stock():
    tickers = {"fields": ["cik", "name", "ticker", "exchange"], "data": [
        [92103, "SOUTHERN CALIFORNIA EDISON Co", "SCE-PG", "NYSE"],
        [1004980, "PG&E Corp", "PCG", "NYSE"],
        [1004980, "PG&E Corp", "PCG-PX", "NYSE"],
    ]}
    filers = {"0000092103": "2026-10-30", "0001004980": "2026-10-30"}
    rows = md.build_company_directory(tickers, filers)
    assert [(r[0], r[3]) for r in rows] == [("PCG", 1), ("PCG-PX", 1), ("SCE-PG", 0)]


def test_unmarked_share_classes_are_all_valued_at_the_listed_price():
    """Warner Music and most SPACs tag one trading symbol without saying which class trades."""
    snapshot = md.build_sec_snapshot(facts_json(industrial_gaap()), {"name": "WARNER MUSIC"})
    snapshot["share_classes"] = [
        {"label": "Class A", "shares_m": 140.0, "symbol": None, "as_of": "2026-10-31"},
        {"label": "Class B", "shares_m": 380.0, "symbol": None, "as_of": "2026-10-31"},
    ]
    data = md.assemble_borrower("WMG", "0000000001", snapshot, price_history(price=30.0), None, None, "n/a")
    assert data["equity_value"]["value"] == pytest.approx(30.0 * 520.0)
    assert any("Every share class (Class A, Class B)" in w for w in data["warnings"])


def test_api_borrower_prices_a_preferred_ticker_through_the_common_stock(offline_sources):
    offline_sources["symbols"] = ["TEST", "TEST-PA"]
    offline_sources["prices"] = price_history(price=25.0)                  # the preferred share
    offline_sources["prices_by_symbol"] = {"TEST": price_history(price=533.16)}
    data = client.get("/api/v1/market/borrower", params={"ticker": "TEST-PA"}).json()
    assert data["ticker"] == "TEST-PA"
    assert data["equity_value"]["value"] == pytest.approx(533.16 * 7_426.1946, rel=1e-6)
    assert any("common stock, TEST" in w for w in data["warnings"])


def test_recent_quarters_wrap_into_the_previous_year():
    from datetime import date
    assert md.recent_quarters(date(2027, 1, 5), 3) == [(2027, 1), (2026, 4), (2026, 3)]


def test_xbrl_index_keeps_each_companys_latest_10k_or_10q():
    text = "\n".join([
        "CIK|Company Name|Form Type|Date Filed|Filename",
        "-----------------------------------------------",
        "789019|MICROSOFT CORP|10-Q|2026-10-28|edgar/data/789019/a.txt",
        "789019|MICROSOFT CORP|10-K|2026-07-30|edgar/data/789019/b.txt",
        "1046179|TAIWAN SEMICONDUCTOR|6-K|2026-10-10|edgar/data/1046179/c.txt",
        "1067983|BERKSHIRE HATHAWAY INC|10-Q/A|2026-11-02|edgar/data/1067983/d.txt",
    ])
    assert md.parse_xbrl_index(text) == {"0000789019": "2026-10-28", "0001067983": "2026-11-02"}


def test_directory_puts_automatic_companies_first_in_sec_order():
    tickers = {"fields": ["cik", "name", "ticker", "exchange"], "data": [
        [1046179, "TAIWAN SEMICONDUCTOR", "TSM", "NYSE"],
        [789019, "MICROSOFT CORP", "MSFT", "Nasdaq"],
        [937966, "ASML HOLDING NV", "ASML", "Nasdaq"],
        [1067983, "BERKSHIRE HATHAWAY INC", "BRK-B", "NYSE"],
    ]}
    filers = {"0000789019": "2026-10-28", "0001067983": "2026-11-02"}
    rows = md.build_company_directory(tickers, filers)
    assert [(r[0], r[3]) for r in rows] == [("MSFT", 1), ("BRK-B", 1), ("TSM", 0), ("ASML", 0)]


def test_api_companies_lists_every_ticker_with_automatic_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_SEC_USER_AGENT", "Aegis Tests tests@example.invalid")
    monkeypatch.setattr(md, "CACHE", md.JsonCache(tmp_path))

    async def fake_json(_client, url, timeout):
        assert url == md.SEC_TICKERS_EXCHANGE_URL
        return {"fields": ["cik", "name", "ticker", "exchange"],
                "data": [[1046179, "TAIWAN SEMICONDUCTOR", "TSM", "NYSE"], [789019, "MICROSOFT CORP", "MSFT", "Nasdaq"]]}

    async def fake_index(_client, url):
        return "789019|MICROSOFT CORP|10-Q|2026-10-28|edgar/data/789019/a.txt" if "QTR" in url else None

    monkeypatch.setattr(md, "_sec_json", fake_json)
    monkeypatch.setattr(md, "_sec_gzip_text", fake_index)
    data = client.get("/api/v1/market/companies").json()
    assert data["fields"] == ["ticker", "name", "exchange", "automatic", "cik"]
    assert data["companies"] == [["MSFT", "MICROSOFT CORP", "Nasdaq", 1, "0000789019"],
                                 ["TSM", "TAIWAN SEMICONDUCTOR", "NYSE", 0, "0001046179"]]
    assert data["automatic_count"] == 1


def test_api_borrower_unknown_ticker_is_404(offline_sources):
    res = client.get("/api/v1/market/borrower", params={"ticker": "NOPE"})
    assert res.status_code == 404
    assert "file with the SEC" in res.json()["detail"]


def test_api_borrower_without_sec_contact_is_503(monkeypatch):
    monkeypatch.delenv("AEGIS_SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(md, "_dotenv_values", lambda: {})
    res = client.get("/api/v1/market/borrower", params={"ticker": "MSFT"})
    assert res.status_code == 503
    assert "AEGIS_SEC_USER_AGENT" in res.json()["detail"]


# --- Banks in the comprehensive assessment and memo ----------------------------------------------

def _bank_like_request(financial_institution):
    """Strong market position but a bank-shaped balance sheet that Altman Z'' reads as distress."""
    return {
        "company_name": "Big Bank",
        "ticker": "BANK",
        "merton_input": {
            "equity_value": 400_000.0, "equity_volatility": 0.25, "short_term_debt": 50_000.0,
            "long_term_debt": 50_000.0, "risk_free_rate": 0.045, "maturity": 1.0, "loss_given_default": 0.45,
        },
        "forensic_input": {
            "sales_t": 180_000.0, "cogs_t": 0.0, "receivables_t": 0.0, "current_assets_t": 0.0, "ppe_t": 0.0,
            "total_assets_t": 4_400_000.0, "depreciation_t": 8_000.0, "sga_t": 0.0, "net_income_t": 57_000.0,
            "cash_flow_operations_t": -140_000.0, "current_liabilities_t": 0.0, "long_term_debt_t": 430_000.0,
            "total_liabilities_t": 4_040_000.0, "inventory_t": 0.0, "accounts_payable_t": 0.0, "ebit_t": 72_000.0,
            "retained_earnings_t": 410_000.0, "book_value_equity_t": 360_000.0,
            "sales_t_minus_1": 175_000.0, "cogs_t_minus_1": 0.0, "receivables_t_minus_1": 0.0,
            "current_assets_t_minus_1": 0.0, "ppe_t_minus_1": 0.0, "total_assets_t_minus_1": 4_000_000.0,
            "depreciation_t_minus_1": 7_500.0, "sga_t_minus_1": 0.0, "long_term_debt_t_minus_1": 400_000.0,
            "total_liabilities_t_minus_1": 3_650_000.0, "inventory_t_minus_1": 0.0, "accounts_payable_t_minus_1": 0.0,
        },
        "is_financial_institution": financial_institution,
        "loan_tenor_years": 5,
    }


def test_bank_band_ignores_the_forensic_models():
    """Main's guard makes the forensic models step aside for financial institutions; the band must
    then rest on distance to default, not read the neutral Altman score of 0 as distress."""
    as_company = client.post("/api/v1/credit/comprehensive", json=_bank_like_request(False)).json()
    as_bank = client.post("/api/v1/credit/comprehensive", json=_bank_like_request(True)).json()
    assert as_company["forensics"]["altman"]["z_double_prime"] < 0.8
    assert as_company["internal_risk_band"].startswith("Band 7")
    assert as_bank["forensics"]["is_financial_institution"] is True
    assert not as_bank["internal_risk_band"].startswith("Band 7")
    assert "not applied" in as_bank["executive_summary"]


def test_bank_memo_says_forensics_were_excluded():
    res = client.post("/api/v1/credit/memo/html", json=_bank_like_request(True))
    assert res.status_code == 200
    assert "excluded from the risk band" in res.text
