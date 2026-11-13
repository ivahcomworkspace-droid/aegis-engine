"""
Writes docs/supported_companies.csv: every ticker in the SEC's company list, with "A" in the first
column when the dashboard's Load button fills it automatically (the company filed a 10-K or 10-Q
in the last three calendar quarters). Rows marked A come first, each group in the SEC's order,
which is roughly by market value.

Needs AEGIS_SEC_USER_AGENT, like Load itself (see .env.example).

    python scripts/list_supported_companies.py
"""

import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.market_data import MarketDataError, company_directory  # noqa: E402


def main() -> int:
    try:
        directory = asyncio.run(company_directory())
    except MarketDataError as exc:
        print(f"Couldn't build the list: {exc.message}", file=sys.stderr)
        return 1

    rows = directory["companies"]
    out = ROOT / "docs" / "supported_companies.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["automatic", "ticker", "company", "exchange", "cik"])
        for ticker, name, exchange, automatic, cik in rows:
            writer.writerow(["A" if automatic else "", ticker, name, exchange, cik])

    automatic = [row for row in rows if row[3]]
    by_exchange = Counter(row[2] or "none" for row in automatic)
    print(f"{len(rows):,} tickers; {len(automatic):,} load automatically ({len({r[4] for r in automatic}):,} companies)")
    print("Filings checked: " + ", ".join(directory["quarters"]))
    print("Automatic by exchange: " + ", ".join(f"{name} {count:,}" for name, count in by_exchange.most_common()))
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
