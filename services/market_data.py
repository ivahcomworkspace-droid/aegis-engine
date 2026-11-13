"""
Live borrower data: fills the Merton inputs and the memo's financial statements from a ticker.

Sources (all free):
- SEC EDGAR (official): ticker -> CIK, XBRL "company facts" (debt, shares outstanding and the
  last two annual reports) and "submissions" (SIC industry code). The SEC asks every automated
  client to identify its operator, so requests need AEGIS_SEC_USER_AGENT, for example
  "Acme Credit risk@acme.com", set in the environment or in the project's .env file.
- US Treasury daily par yield curve (official): the 1-year yield is the risk-free rate.
- Daily prices: a pluggable provider chosen with AEGIS_PRICE_PROVIDER. The default reads Yahoo
  Finance's public chart endpoint, which is free but unofficial and can change or throttle
  without notice. A licensed feed only needs a provider class registered in PRICE_PROVIDERS.

Money values are in $ millions and rates are decimals, the units the engine uses throughout.
Each source fails independently: a missing price or rate is reported as a warning while the
other inputs still load, and the dashboard keeps its manual value for anything not filled.
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import io
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from pydantic import ValidationError

try:
    from aegis_engine.schemas.credit_models import ForensicInput
except ModuleNotFoundError:
    from schemas.credit_models import ForensicInput


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_XBRL_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/xbrl.gz"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
    "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)
YAHOO_CHART_URLS = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1y&interval=1d&includeAdjustedClose=true",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range=1y&interval=1d&includeAdjustedClose=true",
)
TOOL_USER_AGENT = "Mozilla/5.0 (compatible; AegisEngine/2.0)"

TICKERS_TTL = 24 * 3600
DIRECTORY_TTL = 24 * 3600
DIRECTORY_QUARTERS = 3      # a company that filed a 10-K/10-Q this quarter or the two before loads automatically
FILINGS_TTL = 12 * 3600
TREASURY_TTL = 6 * 3600
PRICES_TTL = 15 * 60

TRADING_DAYS = 252
MIN_RETURNS_FOR_VOLATILITY = 60
STALE_PRICE_DAYS = 7
STALE_SHARES_DAYS = 400

# Commercial banks and savings institutions (SEC SIC codes). Any filer whose deposits are at
# least DEPOSIT_SHARE_FOR_BANK of its liabilities is treated the same way.
BANK_SIC_CODES = {"6021", "6022", "6029", "6035", "6036"}
DEPOSIT_SHARE_FOR_BANK = 0.10
# Finance, insurance and real estate. Beneish and Altman built their models on companies outside
# this range, so their scores (and the cash cycle) don't count for these borrowers.
FINANCIAL_SIC_RANGE = range(6000, 7000)

# Listed parents that replaced an older registrant in a holding-company reorganisation, mapped to
# the predecessor whose annual reports (10-K) cover the years before the switch.
PREDECESSOR_CIKS: Dict[str, Tuple[str, str]] = {
    "0002115436": ("0000034088", "Exxon Mobil Corp"),   # ExxonMobil Holdings Corp (XOM), 2026
}
SEC_ARCHIVE_INSTANCE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{stem}_htm.xml"

PERIODIC_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A", "10-Q", "10-Q/A", "10-QT", "10-QT/A"}
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}

# --- XBRL concepts, most specific first ------------------------------------------------------
DEBT_CURRENT_TOTAL = ("DebtCurrent",)
CURRENT_PORTION_LTD = ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent")
# These overlap (commercial paper is usually part of short-term borrowings), so the largest is used
SHORT_TERM_BORROWINGS = (
    "ShortTermBorrowings",
    "CommercialPaper",
    "OtherShortTermBorrowings",
    "NotesPayableCurrent",
    "LinesOfCreditCurrent",
    "ShortTermBankLoansAndNotesPayable",
)
NONCURRENT_LTD = ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "OtherLongTermDebtNoncurrent")
TOTAL_LTD = ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")
BANK_OTHER_LTD = ("AdvancesFromFederalHomeLoanBanks", "SubordinatedDebt", "JuniorSubordinatedNotes")
DEPOSITS = ("Deposits",)
DEPOSIT_PARTS = (
    ("InterestBearingDepositLiabilities", "NoninterestBearingDepositLiabilities"),
    (
        "InterestBearingDepositLiabilitiesDomestic",
        "InterestBearingDepositLiabilitiesForeign",
        "NoninterestBearingDepositLiabilitiesDomestic",
        "NoninterestBearingDepositLiabilitiesForeign",
    ),
)
REPO_TOTAL = ("FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase",)
REPO_PARTS = ("FederalFundsPurchased", "SecuritiesSoldUnderAgreementsToRepurchase")

REVENUE = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",
    "InterestAndDividendIncomeOperating",
)
COST_OF_SALES = (
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfServices",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
)
RECEIVABLES = ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent")
CURRENT_ASSETS = ("AssetsCurrent",)
PPE = (
    "PropertyPlantAndEquipmentNet",
    "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
)
DEPRECIATION = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "Depreciation",
)
SGA = ("SellingGeneralAndAdministrativeExpense",)
SGA_PARTS = (("SellingAndMarketingExpense", "GeneralAndAdministrativeExpense"), ("SellingExpense", "GeneralAndAdministrativeExpense"))
NET_INCOME = ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
OPERATING_CASH_FLOW = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CURRENT_LIABILITIES = ("LiabilitiesCurrent",)
INVENTORY = ("InventoryNet",)
PAYABLES = ("AccountsPayableCurrent", "AccountsPayableTradeCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent")
OPERATING_INCOME = ("OperatingIncomeLoss",)
PRETAX_INCOME = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
)
INTEREST_EXPENSE = ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt")
RETAINED_EARNINGS = ("RetainedEarningsAccumulatedDeficit",)
TOTAL_EQUITY = ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity")
SECURITIES = ("MarketableSecuritiesCurrent", "ShortTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesCurrent")
TOTAL_LIABILITIES = ("Liabilities",)
LIABILITIES_AND_EQUITY = ("LiabilitiesAndStockholdersEquity",)


class MarketDataError(Exception):
    """A failure reported to the user, carrying the HTTP status the API should answer with."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# --- Settings ----------------------------------------------------------------------------------

def _dotenv_values() -> Dict[str, str]:
    """Reads KEY=value lines from the project's .env file (read on each call, so edits apply
    without restarting the server)."""
    try:
        text = (PROJECT_ROOT / ".env").read_text(encoding="utf-8")
    except OSError:
        return {}
    values: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Environment variable first, then the project's .env file."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return _dotenv_values().get(name) or default


def sec_user_agent() -> str:
    agent = setting("AEGIS_SEC_USER_AGENT")
    if not agent or "@" not in agent:
        raise MarketDataError(
            503,
            "Live data isn't set up yet: the SEC requires every automated request to include a contact. "
            "Set AEGIS_SEC_USER_AGENT to your name or company and an email, for example "
            "\"Acme Credit risk@acme.com\", in the environment or in the project's .env file.",
        )
    return agent


# --- Cache -------------------------------------------------------------------------------------

class JsonCache:
    """Time-limited JSON cache kept in memory and mirrored to disk, so restarts and uvicorn
    workers reuse each other's downloads. Disk errors are ignored: the cache is best effort."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._memory: Dict[str, Tuple[float, Any]] = {}

    def _path(self, key: str) -> Path:
        return self.directory / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def get(self, key: str, ttl: float) -> Any:
        now = time.time()
        hit = self._memory.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
        try:
            stored = json.loads(self._path(key).read_text(encoding="utf-8"))
            if now - float(stored["saved_at"]) < ttl:
                self._memory[key] = (float(stored["saved_at"]), stored["value"])
                return stored["value"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def put(self, key: str, value: Any) -> None:
        now = time.time()
        self._memory[key] = (now, value)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self._path(key)
            temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            temp.write_text(json.dumps({"saved_at": now, "value": value}), encoding="utf-8")
            os.replace(temp, target)
        except OSError:
            pass

    def clear(self) -> None:
        self._memory.clear()


CACHE = JsonCache(Path(setting("AEGIS_CACHE_DIR") or PROJECT_ROOT / ".cache" / "market"))


# --- Small helpers -----------------------------------------------------------------------------

def _day(text: str) -> date:
    return date.fromisoformat(text[:10])


def _millions(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value) / 1_000_000.0, 3)


def _fmt_m(value_m: float) -> str:
    """$ millions as a short label, matching the dashboard: $3.96T, $31.1B, $4.39B, $80.0M."""
    size = abs(value_m)
    if size >= 1_000_000:
        return f"${value_m / 1_000_000:,.2f}T"
    if size >= 1_000:
        decimals = 0 if size >= 100_000 else 1 if size >= 10_000 else 2
        return f"${value_m / 1_000:,.{decimals}f}B"
    return f"${value_m:,.1f}M"


def _describe(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    if isinstance(exc, MarketDataError):
        return exc.message
    return str(exc) or exc.__class__.__name__


TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,11}$")


def normalize_ticker(raw: str) -> str:
    """Upper-cases a ticker and writes share classes the SEC's way (BRK.B -> BRK-B)."""
    ticker = (raw or "").strip().upper().replace(".", "-").replace("/", "-")
    if not TICKER_PATTERN.match(ticker):
        raise MarketDataError(422, "Enter a stock ticker such as MSFT or BRK-B.")
    return ticker


# --- SEC company facts -------------------------------------------------------------------------

class CompanyFacts:
    """Lookups over the SEC's XBRL company-facts JSON (non-dimensional facts only)."""

    def __init__(self, raw: Dict[str, Any]):
        facts = raw.get("facts") or {}
        self.name: str = raw.get("entityName") or ""
        self._taxonomies = {"us-gaap": facts.get("us-gaap") or {}, "dei": facts.get("dei") or {}}

    @property
    def has_us_gaap(self) -> bool:
        return bool(self._taxonomies["us-gaap"])

    def rows(self, concept: str, unit: str = "USD", taxonomy: str = "us-gaap",
             forms: Iterable[str] = PERIODIC_FORMS) -> List[Dict[str, Any]]:
        forms = set(forms)
        units = (self._taxonomies[taxonomy].get(concept) or {}).get("units") or {}
        return [r for r in units.get(unit, []) if r.get("form") in forms and "val" in r and "end" in r]

    def instant(self, concept: str, on: str, unit: str = "USD") -> Optional[float]:
        """Balance-sheet value at date `on`, taken from the most recently filed report."""
        rows = [r for r in self.rows(concept, unit) if r["end"] == on and "start" not in r]
        if not rows:
            return None
        return float(max(rows, key=lambda r: r.get("filed", ""))["val"])

    def latest_instant(self, concept: str, on: str, max_age_days: int, unit: str = "USD") -> Optional[Tuple[float, str]]:
        """Most recent balance-sheet value at or before `on`, no older than max_age_days."""
        oldest = _day(on) - timedelta(days=max_age_days)
        rows = [r for r in self.rows(concept, unit) if "start" not in r and oldest <= _day(r["end"]) <= _day(on)]
        if not rows:
            return None
        best = max(rows, key=lambda r: (r["end"], r.get("filed", "")))
        return float(best["val"]), best["end"]

    def annual(self, concept: str, fiscal_year_end: str) -> Optional[float]:
        """Full-year value (about 365 days) ending on fiscal_year_end, from annual reports."""
        matches = []
        for r in self.rows(concept, forms=ANNUAL_FORMS):
            if r["end"] != fiscal_year_end or "start" not in r:
                continue
            if 350 <= (_day(r["end"]) - _day(r["start"])).days <= 380:
                matches.append(r)
        if not matches:
            return None
        return float(max(matches, key=lambda r: r.get("filed", ""))["val"])

    def latest_balance_sheet(self) -> Optional[Dict[str, Any]]:
        """The newest Assets fact in a 10-K or 10-Q: its date, form and filing date."""
        rows = [r for r in self.rows("Assets") if "start" not in r]
        if not rows:
            return None
        return max(rows, key=lambda r: (r["end"], r.get("filed", "")))

    def fiscal_year_ends(self) -> List[str]:
        return sorted({r["end"] for r in self.rows("Assets", forms=ANNUAL_FORMS) if "start" not in r}, reverse=True)

    def first_instant(self, concepts: Sequence[str], on: str) -> Optional[Tuple[str, float]]:
        for concept in concepts:
            value = self.instant(concept, on)
            if value is not None:
                return concept, value
        return None

    def first_annual(self, concepts: Sequence[str], fiscal_year_end: str) -> Optional[float]:
        for concept in concepts:
            value = self.annual(concept, fiscal_year_end)
            if value is not None:
                return value
        return None

    def sum_instant(self, concept_groups: Sequence[Sequence[str]], on: str) -> Optional[Tuple[str, float]]:
        """First group whose concepts are all reported at `on`, summed."""
        for group in concept_groups:
            values = [self.instant(c, on) for c in group]
            if all(v is not None for v in values):
                return " + ".join(group), float(sum(values))
        return None


@dataclass
class DebtComponent:
    term: str          # "short" or "long"
    label: str
    concept: str
    value_m: float
    as_of: str


@dataclass
class DebtFigures:
    short_term_m: float
    long_term_m: float
    as_of: str
    components: List[DebtComponent] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def is_deposit_taker(sic: Optional[str], facts: CompanyFacts, on: str) -> bool:
    """Banks and other deposit takers, whose deposits behave like short-term debt."""
    if sic and str(sic).strip() in BANK_SIC_CODES:
        return True
    deposits = facts.instant("Deposits", on)
    liabilities = facts.instant("Liabilities", on)
    return bool(deposits and liabilities and deposits >= DEPOSIT_SHARE_FOR_BANK * liabilities)


def is_financial_company(sic: Optional[str]) -> bool:
    try:
        return int(str(sic).strip()) in FINANCIAL_SIC_RANGE
    except (TypeError, ValueError):
        return False


def extract_debt(facts: CompanyFacts, on: str, bank: bool) -> DebtFigures:
    """Splits the balance sheet at date `on` into the engine's short-term and long-term debt.

    Companies: debt due within a year (current portion of long-term debt plus short-term
    borrowings) and long-term debt due after a year. Banks: customer deposits, repo funding and
    short-term borrowings count as short-term debt, all long-term debt as long-term (the
    convention used in the SVB case study)."""
    figures = DebtFigures(short_term_m=0.0, long_term_m=0.0, as_of=on)

    def add(term: str, label: str, concept: str, value_m: float, as_of: str = on) -> None:
        figures.components.append(DebtComponent(term, label, concept, round(value_m, 3), as_of))
        if term == "short":
            figures.short_term_m += value_m
        else:
            figures.long_term_m += value_m

    def add_found(term: str, label: str, found: Optional[Tuple[str, float]]) -> None:
        if found:
            add(term, label, found[0], _millions(found[1]) or 0.0)

    def add_largest_short_term_borrowing() -> None:
        found = [(c, v) for c in SHORT_TERM_BORROWINGS if (v := facts.instant(c, on)) is not None]
        if found:
            concept, value = max(found, key=lambda item: item[1])
            if value:
                add("short", "Short-term borrowings", concept, _millions(value) or 0.0)

    current_ltd = facts.first_instant(CURRENT_PORTION_LTD, on)
    noncurrent = facts.first_instant(NONCURRENT_LTD, on)
    total = facts.first_instant(TOTAL_LTD, on)

    if bank:
        deposits = facts.first_instant(DEPOSITS, on) or facts.sum_instant(DEPOSIT_PARTS, on)
        add_found("short", "Customer deposits", deposits)
        add_found("short", "Fed funds purchased and repos", facts.first_instant(REPO_TOTAL, on) or facts.sum_instant([REPO_PARTS], on))
        add_largest_short_term_borrowing()
        if total:
            add_found("long", "Long-term debt", total)
        elif noncurrent:
            add_found("long", "Long-term debt", noncurrent)
            add_found("long", "Long-term debt due within a year", current_ltd)
        else:
            for concept in BANK_OTHER_LTD:
                value = facts.instant(concept, on)
                if value:
                    add("long", "Long-term borrowings", concept, _millions(value) or 0.0)
        if deposits:
            figures.notes.append(
                f"Bank: {_fmt_m(_millions(deposits[1]) or 0.0)} of customer deposits are counted as "
                "short-term debt, as in the SVB case study."
            )
        else:
            figures.notes.append("Bank: no deposit total is tagged in the latest filing, so check short-term debt.")
    else:
        current_total = facts.first_instant(DEBT_CURRENT_TOTAL, on)
        if current_total:
            add_found("short", "Debt due within a year", current_total)
        else:
            add_found("short", "Long-term debt due within a year", current_ltd)
            add_largest_short_term_borrowing()

        if noncurrent:
            add_found("long", "Long-term debt", noncurrent)
        elif total:
            current_m = (_millions(current_ltd[1]) or 0.0) if current_ltd else 0.0
            add("long", "Long-term debt less the part due within a year", total[0],
                max((_millions(total[1]) or 0.0) - current_m, 0.0))
        else:
            for concept in NONCURRENT_LTD:
                older = facts.latest_instant(concept, on, STALE_SHARES_DAYS)
                if older:
                    add("long", "Long-term debt", concept, _millions(older[0]) or 0.0, older[1])
                    figures.notes.append(f"Long-term debt isn't in the latest filing, so the {older[1]} figure is used.")
                    break
            else:
                if figures.components:
                    figures.notes.append("No long-term debt is reported in the latest filing.")

        if not figures.components:
            # Some filers (Ford, GM, Berkshire) report debt only under their own tags, which the SEC's
            # company-facts feed leaves out. Liabilities then stand in for debt, as in KMV's original
            # default point: current liabilities as short-term, the rest as long-term.
            equity = facts.first_instant(TOTAL_EQUITY, on)
            current = facts.instant("LiabilitiesCurrent", on)
            other = facts.instant("LiabilitiesNoncurrent", on)
            total_liabilities = facts.instant("Liabilities", on)
            if total_liabilities is None:
                both = facts.instant("LiabilitiesAndStockholdersEquity", on)
                if both is not None and equity:
                    total_liabilities = both - equity[1]
            if current is not None and (other is not None or total_liabilities is not None):
                add("short", "Current liabilities", "LiabilitiesCurrent", _millions(current) or 0.0)
                if other is None:
                    other = max((total_liabilities or 0.0) - current, 0.0)
                add("long", "Other liabilities", "LiabilitiesNoncurrent", _millions(other) or 0.0)
                figures.notes.append(
                    "Debt isn't tagged in a standard form, so liabilities stand in for it: current liabilities "
                    "as short-term debt and the rest as long-term (KMV's original default point). This "
                    "overstates debt a little, since it includes payables and other obligations."
                )
            elif total_liabilities:
                add("long", "Total liabilities", "Liabilities", _millions(total_liabilities) or 0.0)
                figures.notes.append(
                    "Debt isn't tagged in a standard form and the balance sheet isn't split into current and "
                    "long-term items, so all liabilities count as long-term debt. Raise short-term debt by hand "
                    "if part of it falls due within a year."
                )
            else:
                figures.notes.append("No debt is tagged in the latest filing; if the company borrows, enter its debt by hand.")

    figures.short_term_m = round(figures.short_term_m, 3)
    figures.long_term_m = round(figures.long_term_m, 3)
    return figures


def extract_shares(facts: CompanyFacts, on: str) -> Optional[Tuple[float, str, str]]:
    """Shares outstanding in millions, the date they were counted, and where they came from."""
    rows = facts.rows("EntityCommonStockSharesOutstanding", unit="shares", taxonomy="dei")
    if rows:
        latest_end = max(r["end"] for r in rows)
        if (_day(on) - _day(latest_end)).days <= STALE_SHARES_DAYS:
            same_date = [r for r in rows if r["end"] == latest_end]
            accn = max(same_date, key=lambda r: r.get("filed", "")).get("accn")
            counts = {float(r["val"]) for r in same_date if r.get("accn") == accn}
            return round(sum(counts) / 1_000_000.0, 4), latest_end, f"{same_date[0]['form']} cover page"
    value = facts.instant("CommonStockSharesOutstanding", on, unit="shares")
    if value:
        return round(value / 1_000_000.0, 4), on, "balance sheet"
    diluted = [
        r for r in facts.rows("WeightedAverageNumberOfDilutedSharesOutstanding", unit="shares")
        if "start" in r and (_day(on) - _day(r["end"])).days <= STALE_SHARES_DAYS
    ]
    if diluted:
        best = max(diluted, key=lambda r: (r["end"], r.get("filed", "")))
        return round(float(best["val"]) / 1_000_000.0, 4), best["end"], "diluted weighted average"
    return None


@dataclass
class AnnualFinancials:
    forensic_input: Dict[str, float]
    fiscal_year_end: str
    prior_fiscal_year_end: str
    notes: List[str] = field(default_factory=list)


def extract_financials(facts: CompanyFacts, bank: bool) -> Tuple[Optional[AnnualFinancials], List[str]]:
    """The last two annual reports (10-K) mapped onto the forensic models' inputs, in $M."""
    ends = facts.fiscal_year_ends()
    if len(ends) < 2:
        return None, ["Fewer than two annual reports are filed, so the memo keeps sample financials."]
    t, t1 = ends[0], ends[1]
    if not 300 <= (_day(t) - _day(t1)).days <= 430:
        return None, [f"The annual reports for {t1} and {t} aren't consecutive years, so the memo keeps sample financials."]

    def flow(concepts: Sequence[str], end: str) -> Optional[float]:
        return _millions(facts.first_annual(concepts, end))

    def stock(concepts: Sequence[str], end: str) -> Optional[float]:
        found = facts.first_instant(concepts, end)
        return _millions(found[1]) if found else None

    def sga(end: str) -> Optional[float]:
        total = flow(SGA, end)
        if total is not None:
            return total
        for group in SGA_PARTS:
            parts = [flow((c,), end) for c in group]
            if all(p is not None for p in parts):
                return round(sum(parts), 3)
        return None

    def ebit(end: str) -> Optional[float]:
        operating = flow(OPERATING_INCOME, end)
        if operating is not None:
            return operating
        pretax = flow(PRETAX_INCOME, end)
        if pretax is None:
            return None
        return round(pretax + (flow(INTEREST_EXPENSE, end) or 0.0), 3)

    def liabilities(end: str, equity: Optional[float]) -> Optional[float]:
        reported = stock(TOTAL_LIABILITIES, end)
        if reported is not None:
            return reported
        both = stock(LIABILITIES_AND_EQUITY, end)
        return round(both - equity, 3) if both is not None and equity is not None else None

    notes: List[str] = []
    equity_t = stock(TOTAL_EQUITY, t)
    equity_t1 = stock(TOTAL_EQUITY, t1)
    values: Dict[str, Optional[float]] = {
        "sales_t": flow(REVENUE, t),
        "sales_t_minus_1": flow(REVENUE, t1),
        "total_assets_t": stock(("Assets",), t),
        "total_assets_t_minus_1": stock(("Assets",), t1),
        "total_liabilities_t": liabilities(t, equity_t),
        "total_liabilities_t_minus_1": liabilities(t1, equity_t1),
        "net_income_t": flow(NET_INCOME, t),
        "cash_flow_operations_t": flow(OPERATING_CASH_FLOW, t),
        "ebit_t": ebit(t),
        "retained_earnings_t": stock(RETAINED_EARNINGS, t),
        "book_value_equity_t": equity_t,
    }
    labels = {
        "sales_t": "revenue", "sales_t_minus_1": "prior-year revenue", "total_assets_t": "total assets",
        "total_assets_t_minus_1": "prior-year total assets", "total_liabilities_t": "total liabilities",
        "total_liabilities_t_minus_1": "prior-year total liabilities", "net_income_t": "net income",
        "cash_flow_operations_t": "operating cash flow", "ebit_t": "operating income",
        "retained_earnings_t": "retained earnings", "book_value_equity_t": "shareholders' equity",
    }
    missing = [labels[k] for k, v in values.items() if v is None]
    if missing:
        return None, [f"The annual reports don't tag {', '.join(missing)}, so the memo keeps sample financials."]

    optional_zero = {
        "cogs": COST_OF_SALES, "receivables": RECEIVABLES, "current_assets": CURRENT_ASSETS, "ppe": PPE,
        "inventory": INVENTORY, "accounts_payable": PAYABLES, "securities": SECURITIES,
    }
    for name, concepts in optional_zero.items():
        is_flow = name == "cogs"
        for suffix, end in (("t", t), ("t_minus_1", t1)):
            found = flow(concepts, end) if is_flow else stock(concepts, end)
            values[f"{name}_{suffix}"] = max(found or 0.0, 0.0)
    for suffix, end in (("t", t), ("t_minus_1", t1)):
        values[f"depreciation_{suffix}"] = max(flow(DEPRECIATION, end) or 0.0, 0.0)
        values[f"sga_{suffix}"] = max(sga(end) or 0.0, 0.0)
        values[f"long_term_debt_{suffix}"] = extract_debt(facts, end, bank).long_term_m
    values["current_liabilities_t"] = max(stock(CURRENT_LIABILITIES, t) or 0.0, 0.0)

    # Temporary (mezzanine) equity can leave assets - liabilities - equity off by more than the
    # schema's 5% tolerance; it is then counted with liabilities.
    gap = values["total_assets_t"] - values["total_liabilities_t"] - values["book_value_equity_t"]
    if abs(gap) > 0.04 * values["total_assets_t"]:
        values["total_liabilities_t"] = round(values["total_assets_t"] - values["book_value_equity_t"], 3)
        notes.append("Temporary equity is counted with liabilities so the balance sheet reconciles.")

    if values["sales_t"] <= 0 or values["sales_t_minus_1"] <= 0:
        return None, ["Revenue isn't positive in both annual reports, so the memo keeps sample financials."]

    try:
        validated = ForensicInput(**values)
    except ValidationError as exc:
        first = exc.errors()[0].get("msg", "invalid figures") if exc.errors() else "invalid figures"
        return None, [f"The filed figures don't pass the forensic checks ({first}), so the memo keeps sample financials."]

    return AnnualFinancials(
        forensic_input=validated.model_dump(),
        fiscal_year_end=t,
        prior_fiscal_year_end=t1,
        notes=notes,
    ), []


def build_sec_snapshot(facts_raw: Dict[str, Any], submissions: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the dashboard needs from one company's SEC data, as plain JSON."""
    facts = CompanyFacts(facts_raw)
    name = (submissions or {}).get("name") or facts.name or "This company"
    if not facts.has_us_gaap:
        raise MarketDataError(
            422,
            f"The SEC's data has no US GAAP financial statements for {name} (it may file a 20-F under IFRS, "
            "or report inside its parent company's filings), so it can't be loaded automatically. "
            "Enter its figures by hand.",
        )
    sheet = facts.latest_balance_sheet()
    if sheet is None:
        raise MarketDataError(422, f"No balance sheet was found in {name}'s 10-K or 10-Q filings. Enter its figures by hand.")
    on = sheet["end"]
    sic = (submissions or {}).get("sic") or None
    bank = is_deposit_taker(sic, facts, on)
    debt = extract_debt(facts, on, bank)
    shares = extract_shares(facts, on)
    financials, financial_notes = extract_financials(facts, bank)
    equity = facts.first_instant(TOTAL_EQUITY, on)

    return {
        "name": name,
        "sic": str(sic) if sic else None,
        "sic_description": (submissions or {}).get("sicDescription") or None,
        "is_bank": bank,
        "is_financial": bank or is_financial_company(sic),
        "balance_sheet_date": on,
        "balance_sheet_form": sheet.get("form"),
        "balance_sheet_filed": sheet.get("filed"),
        "short_term_debt": debt.short_term_m,
        "long_term_debt": debt.long_term_m,
        "debt_components": [asdict(c) for c in debt.components],
        "debt_notes": debt.notes,
        "shares_m": shares[0] if shares else None,
        "shares_as_of": shares[1] if shares else None,
        "shares_source": shares[2] if shares else None,
        "book_equity_m": _millions(equity[1]) if equity else None,
        "share_classes": [],
        "financials": asdict(financials) if financials else None,
        "financial_notes": financial_notes,
    }


def _class_label(member: str) -> str:
    """us-gaap:CommonClassAMember -> Class A; v:CommonClassB1Member -> Class B1."""
    local = re.sub(r"Member$", "", member.split(":")[-1])
    match = re.fullmatch(r"(?:Common)?(?:Stock)?Class([A-Z0-9]+)", local)
    if match:
        return f"Class {match.group(1)}"
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", local)


def parse_cover_share_classes(instance_xml: str) -> List[Dict[str, Any]]:
    """Shares outstanding per share class, with each class's ticker, from the cover page of a
    filing's XBRL instance. Companies with several classes (Berkshire, Visa) report their share
    counts this way only, so the company-facts feed has no total for them."""
    contexts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for match in re.finditer(r"<(?:xbrli:)?context\b[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", instance_xml, re.S):
        member = re.search(r"dimension=\"us-gaap:StatementClassOfStockAxis\"[^>]*>\s*([^<\s]+)\s*<", match.group(2))
        instant = re.search(r"<(?:xbrli:)?instant>\s*([^<\s]+)\s*<", match.group(2))
        contexts[match.group(1)] = (member.group(1) if member else None, instant.group(1) if instant else None)

    def dei_facts(name: str) -> List[Tuple[str, str]]:
        pattern = r"<dei:%s\b[^>]*\bcontextRef=\"([^\"]+)\"[^>]*>([^<]*)</dei:%s>" % (name, name)
        return [(ref, value.strip()) for ref, value in re.findall(pattern, instance_xml)]

    symbols: Dict[str, str] = {}
    for ref, value in dei_facts("TradingSymbol"):
        member = contexts.get(ref, (None, None))[0]
        symbol = value.upper().replace(".", "-").replace("/", "-")
        if member and TICKER_PATTERN.match(symbol):
            symbols[member] = symbol

    classes = []
    for ref, value in dei_facts("EntityCommonStockSharesOutstanding"):
        member, instant = contexts.get(ref, (None, None))
        try:
            shares = float(value.replace(",", ""))
        except ValueError:
            continue
        if member and shares > 0:
            classes.append({
                "label": _class_label(member),
                "shares_m": round(shares / 1_000_000.0, 6),
                "symbol": symbols.get(member),
                "as_of": instant,
            })
    return classes


# --- Treasury ----------------------------------------------------------------------------------

def parse_treasury_csv(text: str) -> Optional[Tuple[str, float]]:
    """Latest (date, 1-year par yield as a decimal) in a Treasury daily yield-curve CSV."""
    observations = []
    for row in csv.DictReader(io.StringIO(text)):
        raw = (row.get("1 Yr") or "").strip()
        if not raw:
            continue
        try:
            observed = datetime.strptime((row.get("Date") or "").strip(), "%m/%d/%Y").date()
            observations.append((observed, float(raw) / 100.0))
        except ValueError:
            continue
    if not observations:
        return None
    observed, value = max(observations)
    return observed.isoformat(), value


def par_yield_to_continuous(par_yield: float) -> float:
    """Treasury curve yields are bond-equivalent (semi-annual); the engine wants continuous."""
    return 2.0 * math.log1p(par_yield / 2.0)


async def treasury_one_year(client: httpx.AsyncClient) -> Tuple[str, float]:
    cached = CACHE.get("treasury_1y", TREASURY_TTL)
    if cached:
        return cached[0], float(cached[1])
    year = datetime.now(timezone.utc).year
    for candidate in (year, year - 1):          # early January has no rows for the new year yet
        response = await client.get(
            TREASURY_CSV_URL.format(year=candidate), headers={"User-Agent": TOOL_USER_AGENT}, timeout=15.0
        )
        response.raise_for_status()
        parsed = parse_treasury_csv(response.text)
        if parsed:
            CACHE.put("treasury_1y", list(parsed))
            return parsed
    raise RuntimeError("no 1-year yield published")


# --- Prices ------------------------------------------------------------------------------------

@dataclass
class PriceHistory:
    symbol: str
    currency: str
    last_price: float
    last_date: str
    closes: List[float]
    display_name: Optional[str]
    source: str


def parse_yahoo_chart(payload: Dict[str, Any], source: str) -> PriceHistory:
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        error = (chart.get("error") or {}).get("description") or "no data"
        raise ValueError(error)
    result = results[0]
    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    adjusted = ((indicators.get("adjclose") or [{}])[0] or {}).get("adjclose") or []
    closes_raw = ((indicators.get("quote") or [{}])[0] or {}).get("close") or []
    series = adjusted if any(adjusted) else closes_raw
    closes = [float(c) for c in series if c is not None and c > 0]
    price = meta.get("regularMarketPrice") or next((c for c in reversed(closes_raw) if c), None)
    if not price:
        raise ValueError("no price")
    stamp = meta.get("regularMarketTime") or ((result.get("timestamp") or [None])[-1])
    last_date = datetime.fromtimestamp(stamp, timezone.utc).date().isoformat() if stamp else ""
    return PriceHistory(
        symbol=meta.get("symbol") or "",
        currency=meta.get("currency") or "",
        last_price=float(price),
        last_date=last_date,
        closes=closes,
        display_name=meta.get("longName") or meta.get("shortName"),
        source=source,
    )


def annualized_volatility(closes: Sequence[float], window: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of daily log returns over the last `window` trading days."""
    recent = list(closes)[-(window + 1):]
    returns = [math.log(b / a) for a, b in zip(recent, recent[1:]) if a > 0 and b > 0]
    if len(returns) < MIN_RETURNS_FOR_VOLATILITY:
        raise ValueError(f"only {len(returns)} daily returns")
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance * TRADING_DAYS)


class YahooChartProvider:
    """Free, unofficial Yahoo Finance chart endpoint (no key; may throttle or change)."""

    name = "yahoo"
    label = "Yahoo Finance (unofficial)"

    async def history(self, client: httpx.AsyncClient, symbol: str) -> PriceHistory:
        last_error: Optional[BaseException] = None
        for url in YAHOO_CHART_URLS:
            try:
                response = await client.get(
                    url.format(symbol=symbol),
                    headers={"User-Agent": TOOL_USER_AGENT, "Accept": "application/json"},
                    timeout=10.0,
                )
                response.raise_for_status()
                return parse_yahoo_chart(response.json(), self.label)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
        raise RuntimeError(_describe(last_error) if last_error else "unavailable")


# A licensed feed plugs in here: a class with `name`, `label` and `async history(client, symbol)`.
PRICE_PROVIDERS = {YahooChartProvider.name: YahooChartProvider}


def price_provider():
    name = (setting("AEGIS_PRICE_PROVIDER") or YahooChartProvider.name).strip().lower()
    provider = PRICE_PROVIDERS.get(name)
    if provider is None:
        raise RuntimeError(f"unknown AEGIS_PRICE_PROVIDER '{name}' (available: {', '.join(PRICE_PROVIDERS)})")
    return provider()


async def price_history(client: httpx.AsyncClient, symbol: str) -> PriceHistory:
    provider = price_provider()
    key = f"prices_{provider.name}_{symbol}"
    cached = CACHE.get(key, PRICES_TTL)
    if cached:
        return PriceHistory(**cached)
    history = await provider.history(client, symbol)
    CACHE.put(key, asdict(history))
    return history


# --- SEC requests ------------------------------------------------------------------------------

async def _sec_json(client: httpx.AsyncClient, url: str, timeout: float) -> Optional[Dict[str, Any]]:
    response = await client.get(url, headers={"User-Agent": sec_user_agent()}, timeout=timeout)
    if response.status_code == 403:
        raise MarketDataError(
            502,
            "The SEC turned the request away. Check that AEGIS_SEC_USER_AGENT holds a real name and email, "
            "then try again in a few minutes.",
        )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


# Preferred shares (-PA, .PRI, TDSPRV), warrants (-WT), units (-UN), rights (-RT, -RI) and
# contingent value rights (-CVR)
NON_COMMON_SUFFIX = re.compile(r"(-(P[A-Z]*|WTS?|WS|UN?|RT?|RI|CVR|W)$)|(\.PR[A-Z]?$)")
SHARE_CLASS_SUFFIX = re.compile(r"-[A-EK]$")          # BRK-A, HEI-A, LEN-B are common share classes


def common_tickers(tickers: Sequence[str]) -> List[str]:
    """The tickers of a company's common stock, in the SEC's order. Leaves out preferred shares,
    warrants, units, rights and notes, which carry a suffix or extend another of the company's
    symbols by a letter or two (RILY -> RILYL, BFRG -> BFRGW). The SEC lists the main common
    ticker first, so it always counts (GOOGL stays, though GOOG is a prefix of it)."""
    plain = {t: t.replace("-", "").replace(".", "") for t in tickers}
    commons = []
    for position, ticker in enumerate(tickers):
        if NON_COMMON_SUFFIX.search(ticker) or (len(ticker) >= 6 and re.search(r"PR[A-Z]$", ticker)):
            continue
        if position > 0 and not SHARE_CLASS_SUFFIX.search(ticker):
            base = plain[ticker]
            if any(other != ticker and base != plain[other] and base.startswith(plain[other])
                   and len(base) - len(plain[other]) <= 2 for other in tickers):
                continue
        commons.append(ticker)
    return commons


def recent_quarters(today: date, count: int = DIRECTORY_QUARTERS) -> List[Tuple[int, int]]:
    """(year, quarter) for the current calendar quarter and the ones before it, newest first."""
    year, quarter = today.year, (today.month - 1) // 3 + 1
    quarters = []
    for _ in range(count):
        quarters.append((year, quarter))
        year, quarter = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
    return quarters


def parse_xbrl_index(text: str) -> Dict[str, str]:
    """CIK (10 digits) -> latest 10-K/10-Q filing date, from an EDGAR quarterly XBRL index."""
    filers: Dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5 or parts[2].strip() not in PERIODIC_FORMS or not parts[0].strip().isdigit():
            continue
        cik, filed = parts[0].strip().zfill(10), parts[3].strip()
        if filed > filers.get(cik, ""):
            filers[cik] = filed
    return filers


def build_company_directory(tickers_exchange: Dict[str, Any], periodic_filers: Dict[str, str]) -> List[List[Any]]:
    """[ticker, name, exchange, automatic (1 or 0), cik] for every SEC ticker. A ticker loads
    automatically when its company files 10-K/10-Q reports and has listed common stock to price
    (a preferred share or warrant is priced through it). Those come first; each group keeps the
    SEC's order (about market value)."""
    fields = tickers_exchange.get("fields") or ["cik", "name", "ticker", "exchange"]
    column = {name: i for i, name in enumerate(fields)}
    records = []
    for rank, record in enumerate(tickers_exchange.get("data") or []):
        try:
            cik = str(record[column["cik"]]).zfill(10)
            ticker = str(record[column["ticker"]]).strip().upper()
            name = str(record[column["name"]]).strip()
        except (IndexError, KeyError, TypeError):
            continue
        if ticker and name:
            exchange = record[column["exchange"]] if "exchange" in column else None
            records.append((rank, cik, ticker, name, exchange or ""))
    by_company: Dict[str, List[str]] = {}
    for _rank, cik, ticker, _name, _exchange in records:
        by_company.setdefault(cik, []).append(ticker)
    has_common = {cik: bool(common_tickers(tickers)) for cik, tickers in by_company.items()}
    ranked = []
    for rank, cik, ticker, name, exchange in records:
        automatic = 1 if cik in periodic_filers and has_common[cik] else 0
        ranked.append((-automatic, rank, [ticker, name, exchange, automatic, cik]))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


async def _sec_gzip_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    response = await client.get(url, headers={"User-Agent": sec_user_agent()}, timeout=45.0)
    if response.status_code == 404:
        return None
    if response.status_code == 403:
        raise MarketDataError(502, "The SEC turned the request away. Check AEGIS_SEC_USER_AGENT and try again in a few minutes.")
    response.raise_for_status()
    try:
        return gzip.decompress(response.content).decode("latin-1")
    except OSError:                                    # already decompressed in transit
        return response.content.decode("latin-1")


async def company_directory(client: Optional[httpx.AsyncClient] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """Every ticker in the SEC's company list, marked `automatic` when its company filed a 10-K or
    10-Q in the last DIRECTORY_QUARTERS quarters (the reports Load reads). Cached for a day."""
    cached = CACHE.get("company_directory", DIRECTORY_TTL)
    if cached is not None:
        return cached
    sec_user_agent()
    own_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    quarters = recent_quarters(today or datetime.now(timezone.utc).date())
    try:
        results = await asyncio.gather(
            _sec_json(client, SEC_TICKERS_EXCHANGE_URL, timeout=30.0),
            *(_sec_gzip_text(client, SEC_XBRL_INDEX_URL.format(year=y, quarter=q)) for y, q in quarters),
            return_exceptions=True,
        )
    finally:
        if own_client:
            await client.aclose()
    tickers = results[0]
    if isinstance(tickers, MarketDataError):
        raise tickers
    if isinstance(tickers, BaseException) or not tickers:
        raise MarketDataError(502, "Couldn't fetch the SEC's company list. Try again shortly.")
    filers: Dict[str, str] = {}
    for index_text in results[1:]:
        if isinstance(index_text, str):
            for cik, filed in parse_xbrl_index(index_text).items():
                if filed > filers.get(cik, ""):
                    filers[cik] = filed
    if not filers:
        raise MarketDataError(502, "Couldn't read the SEC's filing index. Try again shortly.")
    companies = build_company_directory(tickers, filers)
    directory = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quarters": [f"{y} Q{q}" for y, q in quarters],
        "fields": ["ticker", "name", "exchange", "automatic", "cik"],
        "automatic_count": sum(row[3] for row in companies),
        "companies": companies,
    }
    CACHE.put("company_directory", directory)
    return directory


async def resolve_ticker(client: httpx.AsyncClient, ticker: str) -> Tuple[str, str, List[str]]:
    """SEC CIK (10 digits), registrant title, and all of that company's tickers in the SEC's order."""
    mapping = CACHE.get("sec_company_tickers", TICKERS_TTL)
    if mapping is None:
        raw = await _sec_json(client, SEC_TICKERS_URL, timeout=20.0) or {}
        mapping = {str(row["ticker"]).upper(): [str(row["cik_str"]).zfill(10), row.get("title", "")] for row in raw.values()}
        if not mapping:
            raise MarketDataError(502, "The SEC's ticker list came back empty. Try again shortly.")
        CACHE.put("sec_company_tickers", mapping)
    hit = mapping.get(ticker)
    if hit is None:
        raise MarketDataError(
            404,
            f"{ticker} isn't among the companies that file with the SEC, so it can't be loaded automatically. "
            "Enter its figures by hand.",
        )
    return hit[0], hit[1], [symbol for symbol, (cik, _title) in mapping.items() if cik == hit[0]]


async def sec_snapshot(client: httpx.AsyncClient, cik: str) -> Dict[str, Any]:
    key = f"sec_snapshot_{cik}"
    cached = CACHE.get(key, FILINGS_TTL)
    if cached is not None:
        return cached
    facts_raw, submissions = await asyncio.gather(
        _sec_json(client, SEC_FACTS_URL.format(cik=cik), timeout=45.0),
        _sec_json(client, SEC_SUBMISSIONS_URL.format(cik=cik), timeout=20.0),
    )
    if not facts_raw:
        raise MarketDataError(422, "The SEC has no XBRL financial data for this company. Enter its figures by hand.")
    snapshot = build_sec_snapshot(facts_raw, submissions or {})
    if snapshot["shares_m"] is None:
        snapshot["share_classes"] = await cover_share_classes(client, cik, submissions or {})
    if snapshot["financials"] is None and cik in PREDECESSOR_CIKS:
        await _add_predecessor_financials(client, snapshot, *PREDECESSOR_CIKS[cik])
    CACHE.put(key, snapshot)
    return snapshot


async def cover_share_classes(client: httpx.AsyncClient, cik: str, submissions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Share classes from the latest 10-Q/10-K cover page; empty if the filing can't be read."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    for index, form in enumerate(recent.get("form") or []):
        if form in PERIODIC_FORMS:
            accession = str(recent["accessionNumber"][index]).replace("-", "")
            stem = str(recent["primaryDocument"][index]).rsplit(".", 1)[0]
            break
    else:
        return []
    url = SEC_ARCHIVE_INSTANCE_URL.format(cik=int(cik), accession=accession, stem=stem)
    try:
        response = await client.get(url, headers={"User-Agent": sec_user_agent()}, timeout=45.0)
        if response.status_code != 200:
            return []
        return parse_cover_share_classes(response.text)
    except httpx.HTTPError:
        return []


async def _add_predecessor_financials(client: httpx.AsyncClient, snapshot: Dict[str, Any],
                                      predecessor_cik: str, predecessor_name: str) -> None:
    """Fills the memo's annual reports from the registrant this company replaced."""
    try:
        raw = await _sec_json(client, SEC_FACTS_URL.format(cik=predecessor_cik), timeout=45.0)
    except (httpx.HTTPError, MarketDataError):
        return
    financials, _notes = extract_financials(CompanyFacts(raw or {}), snapshot["is_bank"])
    if financials is None:
        return
    financials.notes.append(
        f"The new holding company hasn't filed two annual reports yet, so they come from "
        f"{predecessor_name} (CIK {predecessor_cik}), the listed parent it replaced."
    )
    snapshot["financials"] = asdict(financials)
    snapshot["financial_notes"] = []


# --- Assembly ----------------------------------------------------------------------------------

def _point(value: Optional[float], source: str, as_of: Optional[str], detail: Optional[str] = None) -> Dict[str, Any]:
    return {"value": value, "source": source, "as_of": as_of, "detail": detail}


def assemble_borrower(
    ticker: str,
    cik: str,
    snapshot: Dict[str, Any],
    prices: Optional[PriceHistory],
    price_error: Optional[str],
    rate: Optional[Tuple[str, float]],
    rate_error: Optional[str],
    today: Optional[date] = None,
    class_prices: Optional[Dict[str, PriceHistory]] = None,
    price_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Combines the sources into the dashboard's five inputs, each with its provenance.
    prices belong to price_symbol (the common stock, which differs from `ticker` when a preferred
    share or warrant was asked for); class_prices holds other listed share classes."""
    today = today or datetime.now(timezone.utc).date()
    price_symbol = price_symbol or ticker
    warnings: List[str] = list(snapshot.get("debt_notes") or [])
    sheet_label = f"SEC {snapshot.get('balance_sheet_form') or 'filing'} balance sheet"
    class_prices = dict(class_prices or {})
    if prices is not None:
        class_prices.setdefault(price_symbol, prices)
    if price_symbol != ticker:
        warnings.append(
            f"{ticker} isn't common stock (it's a preferred share, warrant, unit or note), so equity value and "
            f"volatility come from the company's common stock, {price_symbol}."
        )

    equity = _point(None, "Price × shares outstanding", None)
    volatility = _point(None, "Daily closing prices", None)
    if prices is None:
        warnings.append(f"Market prices are unavailable ({price_error}); enter equity value and volatility by hand.")
    else:
        if prices.last_date and (today - _day(prices.last_date)).days > STALE_PRICE_DAYS:
            warnings.append(f"The latest price is from {prices.last_date}; check that {price_symbol} still trades.")
        shares_m = snapshot.get("shares_m")
        classes = snapshot.get("share_classes") or []
        listed = [c for c in classes if c.get("symbol")]
        cap_m: Optional[float] = None
        if prices.currency and prices.currency.upper() != "USD":
            warnings.append(f"{price_symbol} is quoted in {prices.currency}, so equity value isn't filled; enter it in USD.")
        elif shares_m:
            cap_m = round(prices.last_price * shares_m, 3)
            equity = _point(
                cap_m,
                f"{prices.source} price × SEC shares",
                prices.last_date,
                f"${prices.last_price:,.2f} × {shares_m:,.1f}M shares ({snapshot.get('shares_source')}, {snapshot.get('shares_as_of')})",
            )
        elif listed:
            # Several share classes: each listed class at its own price
            unpriced = [c["symbol"] for c in listed if c["symbol"] not in class_prices]
            if unpriced:
                warnings.append(f"No price was found for {', '.join(unpriced)}, so equity value isn't filled; enter it by hand.")
            else:
                cap_m = round(sum(c["shares_m"] * class_prices[c["symbol"]].last_price for c in listed), 3)
                parts = " + ".join(
                    f"{c['symbol']} ${class_prices[c['symbol']].last_price:,.2f} × {c['shares_m']:,.2f}M" for c in listed
                )
                equity = _point(
                    cap_m,
                    f"{prices.source} prices × SEC shares by class",
                    prices.last_date,
                    f"{parts} shares (cover page, {listed[0].get('as_of')})",
                )
                unlisted = [c["label"] for c in classes if not c.get("symbol")]
                if unlisted:
                    names = ", ".join(unlisted[:-1]) + (" and " if len(unlisted) > 1 else "") + unlisted[-1]
                    warnings.append(
                        f"{names} shares aren't listed, so they're left out of equity value. They convert into "
                        "listed shares, so equity value is somewhat understated."
                    )
        elif classes:
            # The filing names one trading symbol without saying which class it belongs to
            # (Warner Music, most SPACs): every class is valued at that price, as market data
            # vendors do for classes that convert one for one.
            total_m = sum(c["shares_m"] for c in classes)
            cap_m = round(prices.last_price * total_m, 3)
            labels = ", ".join(c["label"] for c in classes)
            equity = _point(
                cap_m,
                f"{prices.source} price × SEC shares, all classes",
                prices.last_date,
                f"${prices.last_price:,.2f} × {total_m:,.1f}M shares ({labels}; cover page, {classes[0].get('as_of')})",
            )
            warnings.append(
                f"Every share class ({labels}) is valued at the {price_symbol} price, since the filing doesn't say "
                "which class trades. Adjust equity value if a class converts at a different rate."
            )
        else:
            warnings.append("Shares outstanding aren't tagged in the filings, so equity value isn't filled.")
        book = snapshot.get("book_equity_m")
        if cap_m and book and book > 0 and not 0.05 <= cap_m / book <= 200:
            warnings.append(
                "Market value is far from book value; if the company has several share classes, check the share count."
            )
        try:
            sigma = annualized_volatility(prices.closes)
            volatility = _point(
                round(sigma, 4),
                f"{prices.source}, 1-year daily returns",
                prices.last_date,
                f"{min(len(prices.closes) - 1, TRADING_DAYS)} trading days, annualised",
            )
        except ValueError as exc:
            warnings.append(f"Not enough price history for volatility ({exc}); enter it by hand.")

    components = snapshot.get("debt_components") or []

    def debt_detail(term: str) -> str:
        parts = [f"{c['label']} {_fmt_m(c['value_m'])}" for c in components if c["term"] == term]
        return " + ".join(parts) if parts else "None reported"

    if rate is None:
        warnings.append(f"The Treasury yield is unavailable ({rate_error}); the risk-free rate keeps its current value.")
        risk_free = _point(None, "US Treasury 1-year yield", None)
    else:
        observed, par = rate
        risk_free = _point(
            round(par_yield_to_continuous(par), 6),
            "US Treasury 1-year yield",
            observed,
            f"{par * 100:.2f}% par yield, continuously compounded",
        )

    financials = snapshot.get("financials")
    warnings.extend(snapshot.get("financial_notes") or [])
    if financials:
        warnings.extend(financials.get("notes") or [])
    financial = bool(snapshot.get("is_financial", snapshot.get("is_bank")))
    if financial and financials:
        kind = "bank" if snapshot.get("is_bank") else "financial company"
        warnings.append(
            f"Forensic scores (Beneish, Altman, cash cycle) are built for non-financial companies, so they don't "
            f"count toward this {kind}'s rating."
        )
    elif financials and not financials["forensic_input"].get("cogs_t"):
        warnings.append("No cost of sales is reported, so the cash cycle counts receivable days only.")

    display_name = (prices.display_name if prices else None) or snapshot.get("name") or ticker
    return {
        "ticker": ticker,
        "company_name": display_name,
        "sec_name": snapshot.get("name"),
        "cik": cik,
        "sic": snapshot.get("sic"),
        "sic_description": snapshot.get("sic_description"),
        "is_bank": bool(snapshot.get("is_bank")),
        "equity_value": equity,
        "equity_volatility": volatility,
        "short_term_debt": _point(snapshot["short_term_debt"], sheet_label, snapshot["balance_sheet_date"], debt_detail("short")),
        "long_term_debt": _point(snapshot["long_term_debt"], sheet_label, snapshot["balance_sheet_date"], debt_detail("long")),
        "risk_free_rate": risk_free,
        "forensic_input": financials["forensic_input"] if financials else None,
        "forensics_applicable": not financial,
        "fiscal_year_end": financials["fiscal_year_end"] if financials else None,
        "prior_fiscal_year_end": financials["prior_fiscal_year_end"] if financials else None,
        "warnings": warnings,
        "loaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


async def load_borrower(raw_ticker: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    """Fetches and combines SEC filings, Treasury yields and prices for one ticker."""
    ticker = normalize_ticker(raw_ticker)
    sec_user_agent()                              # fail fast with setup guidance
    own_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    try:
        cik, _title, company_symbols = await resolve_ticker(client, ticker)
        commons = common_tickers(company_symbols)
        price_symbol = ticker if ticker in commons or not company_symbols else (commons[0] if commons else None)

        async def no_common_stock() -> PriceHistory:
            raise RuntimeError(f"{ticker} isn't common stock and the company has no listed common stock to price")

        snapshot, prices, rate = await asyncio.gather(
            sec_snapshot(client, cik),
            price_history(client, price_symbol) if price_symbol else no_common_stock(),
            treasury_one_year(client),
            return_exceptions=True,
        )
        if isinstance(snapshot, MarketDataError):
            raise snapshot
        if isinstance(snapshot, BaseException):
            raise MarketDataError(502, f"Couldn't read {ticker}'s SEC filings ({_describe(snapshot)}). Try again shortly.")

        # Other listed share classes of the same company (BRK-A when BRK-B was typed)
        siblings = sorted({c["symbol"] for c in snapshot.get("share_classes") or [] if c.get("symbol") and c["symbol"] != price_symbol})
        sibling_prices = await asyncio.gather(*(price_history(client, s) for s in siblings), return_exceptions=True)
        class_prices = {s: p for s, p in zip(siblings, sibling_prices) if isinstance(p, PriceHistory)}

        return assemble_borrower(
            ticker,
            cik,
            snapshot,
            prices if isinstance(prices, PriceHistory) else None,
            _describe(prices) if isinstance(prices, BaseException) else None,
            rate if isinstance(rate, tuple) else None,
            _describe(rate) if isinstance(rate, BaseException) else None,
            class_prices=class_prices,
            price_symbol=price_symbol or ticker,
        )
    except httpx.HTTPError as exc:
        raise MarketDataError(502, f"Couldn't reach the SEC ({_describe(exc)}). Try again shortly.") from exc
    finally:
        if own_client:
            await client.aclose()
