# Aegis // Institutional Structural Credit Risk & Forensics Engine

[![Live Web Terminal](https://img.shields.io/badge/Live%20Demo-Render-00c7b7?style=for-the-badge&logo=render&logoColor=white)](https://aegis-engine-s96v.onrender.com)
[![OpenAPI Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://aegis-engine-s96v.onrender.com/docs)
[![Pytest Suite](https://img.shields.io/badge/Tests-123%20Passed-brightgreen?style=for-the-badge)](tests/)

> **Live Production Terminal:** [https://aegis-engine-s96v.onrender.com](https://aegis-engine-s96v.onrender.com)  
> **Interactive Swagger API:** [https://aegis-engine-s96v.onrender.com/docs](https://aegis-engine-s96v.onrender.com/docs)

---

## 1. Executive Summary & Strategic Identity

**Aegis** is an institutional-grade quantitative credit risk, balance-sheet forensics, and loan default probability engine. It serves as the defensive counterpart to equity valuation platforms like Quilet Terminal 2.0:

- **Quilet Terminal (The Offense):** Solves for upside valuation and going-concern enterprise value via 5-year explicit FCFF schedules, dynamic CAPM WACC hurdle rates, reverse DCF expectation inversion, and DuPont return decomposition.
- **Aegis Engine (The Defense):** Solves for asymmetric downside risk, debt service capacity, distance-to-default, latent asset volatility inversion, forensic earnings manipulation, and regulatory expected credit losses (IFRS 9 / CECL / Basel IRB).

---

## 2. Core Mathematical Specifications

### A. The Structural Merton Option Engine (`core/merton.py`)

Model corporate equity as a European call option on latent enterprise assets $V_A$ against a debt default barrier $D = \text{Short-Term Debt} + 0.5 \times \text{Long-Term Debt}$ maturing at $T$:

$$
E = V_A \Phi(d_1) - D e^{-r T} \Phi(d_2)
$$

$$
d_1 = \frac{\ln(V_A / D) + \left(r + 0.5 \sigma_A^2\right)T}{\sigma_A \sqrt{T}}, \quad d_2 = d_1 - \sigma_A \sqrt{T}
$$

Link observable equity volatility $\sigma_E$ to unobservable asset volatility $\sigma_A$ using Itô's Lemma:

$$
\sigma_E = \left(\frac{V_A}{E}\right) \Phi(d_1) \sigma_A
$$

Solve this coupled 2D non-linear system numerically using SciPy's modified Powell hybrid method (`scipy.optimize.root(..., method='hybr')`) with an exact analytical Jacobian matrix $\mathbf{J}$ for sub-2ms convergence:

$$
\mathbf{J} = \begin{bmatrix} \frac{\partial F_1}{\partial V_A} & \frac{\partial F_1}{\partial \sigma_A} \\ \frac{\partial F_2}{\partial V_A} & \frac{\partial F_2}{\partial \sigma_A} \end{bmatrix}
$$

Where:
- $\frac{\partial F_1}{\partial V_A} = \Phi(d_1)$
- $\frac{\partial F_1}{\partial \sigma_A} = V_A \sqrt{T} \phi(d_1)$
- $\frac{\partial F_2}{\partial V_A} = \frac{\sigma_A}{E} \left[ \Phi(d_1) + \frac{\phi(d_1)}{\sigma_A \sqrt{T}} \right]$
- $\frac{\partial F_2}{\partial \sigma_A} = \frac{V_A}{E} \left[ \Phi(d_1) + \sigma_A \phi(d_1) \left( 0.5\sqrt{T} - \frac{\ln(V_A/D) + r T}{\sigma_A^2 \sqrt{T}} \right) \right]$

**Derived Solvency Metrics:**
- **Physical Distance-to-Default:**

$$
\text{DD}_{\text{physical}} = \frac{\ln(V_A / D) + (\mu_A - 0.5\sigma_A^2)T}{\sigma_A \sqrt{T}}, \quad \text{PD}_{\text{physical}} = \Phi(-\text{DD})
$$

- **Risk-Neutral Implied Spread:**

$$
\text{DD}_{\text{RN}} = \frac{\ln(V_A / D) + (r - 0.5\sigma_A^2)T}{\sigma_A \sqrt{T}}
$$

$$
\text{Spread (bps)} = -\frac{\ln(1 - \text{PD}_{\text{RN}} \times \text{LGD})}{T} \times 10{,}000
$$

- **Asymptotic Clamp:** If $D = 0 \implies V_A = E$, $\sigma_A = \sigma_E$, $\text{DD} = 99.0$, $\text{PD} = 0.0$, $\text{Spread} = 0.0\text{ bps}$.

---

### B. Balance Sheet Forensics (`core/forensics.py`)

1. **8-Factor Beneish M-Score:**

$$
\text{M} = -4.84 + 0.920\text{DSRI} + 0.528\text{GMI} + 0.404\text{AQI} + 0.892\text{SGI} + 0.115\text{DEPI} - 0.172\text{SGAI} + 4.679\text{TATA} - 0.327\text{LVGI}
$$

   - Active manipulation threshold: $\text{M-Score} > -1.78$.

2. **Altman $Z''$-Score (Services & Emerging Markets):**

$$
Z'' = 6.56 X_1 + 3.26 X_2 + 6.72 X_3 + 1.05 X_4
$$

   - Solvency zones: Safe ($Z'' > 2.60$), Grey ($1.10 \le Z'' \le 2.60$), Distress ($Z'' < 1.10$).

3. **Cash Conversion Cycle (CCC):**

$$
\text{CCC} = \text{DSO} + \text{DIO} - \text{DPO}
$$

   - Alert triggered when $\Delta\text{CCC} > +20\text{ days}$ while gross margin is static ($|\Delta\text{GM}| \le 0.02$).

---

### C. Regulatory Loss Distribution & Stress Testing (`core/vasicek.py`)

1. **Vasicek ASRF Conditional Distribution:**

$$
\text{PD}_{\text{conditional}}(Z) = \Phi\left( \frac{\Phi^{-1}(\text{PD}_{\text{PIT}}) + \sqrt{\rho} \cdot Z}{\sqrt{1 - \rho}} \right)
$$

   Where corporate asset correlation $\rho$ follows the Basel corporate and SME specification:

$$
\rho = 0.12 \left(\frac{1 - e^{-50 \cdot \text{PD}}}{1 - e^{-50}}\right) + 0.24 \left(1 - \frac{1 - e^{-50 \cdot \text{PD}}}{1 - e^{-50}}\right) - 0.04 \left(1 - \frac{\min(\max(S, 5), 50) - 5}{45}\right)
$$

   With correlation bounded by $\rho \in [0.08, 0.24]$ when SME turnover adjustment is applied ($S \le 50\text{M EUR}$), and $\rho \in [0.12, 0.24]$ for standard corporate exposures.

2. **Multi-Period Lifetime Expected Credit Loss (ECL):**

$$
\text{ECL} = \sum_{t=1}^T \frac{\text{PD}_t^* \times \text{LGD}_t \times \text{EAD}_t}{(1 + \text{EIR})^t}
$$

   - **Stage 1 (Performing):** 12-Month ECL ($T = 1$).
   - **Stage 2 (Underperforming / SICR):** Lifetime ECL triggered when $\Delta\text{DD} < -1.0\sigma$, $\text{DPD} \in [30, 89]$, or $\text{DD} < 2.0\sigma$ (watchlist backstop, matching the Band 5 surveillance stance).
   - **Stage 3 (Credit Impaired):** Lifetime ECL with workout LGD on $\text{DPD} \ge 90$ or default flag.

---

## 3. Engineering Architecture

```
aegis_engine/
├── core/
│   ├── __init__.py
│   ├── merton.py             # SciPy Powell solver ('hybr') + Analytical Jacobian
│   ├── forensics.py          # Beneish 8-Factor, Altman Z'', Cash Conversion Cycle
│   └── vasicek.py            # Vasicek ASRF macro loss engine + Multi-period ECL
├── schemas/
│   ├── __init__.py
│   └── credit_models.py      # Pydantic v2 schemas enforcing balance sheet identities
├── services/
│   ├── __init__.py
│   ├── memo_generator.py     # Credit memo: vector charts, WeasyPrint PDF (ReportLab fallback)
│   ├── formatting.py         # Money and PD formats shared with the dashboard
│   └── market_data.py        # Live borrower data: SEC EDGAR, US Treasury yields, price feed
├── api/
│   ├── __init__.py
│   └── main.py               # Asynchronous FastAPI microservice & REST endpoints
├── frontend/
│   └── index.html            # Responsive single-page web terminal (UrbanHome theme)
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest root configuration
│   ├── test_engine.py        # 77 deterministic unit, numerical & integration tests
│   ├── test_market_data.py   # 38 offline tests for live borrower data, company list and financial-sector handling
│   └── test_memo.py          # 8 tests: the memo prints the dashboard's figures, formats and sources
├── .env.example              # Settings for live data (copy to .env, which git ignores)
├── Dockerfile                # Multi-stage production container build (python:3.11-slim)
├── requirements.txt          # Frozen production dependencies
└── Makefile                  # Unified developer CLI: test, lint, run, docker-build
```

---

## 4. API Endpoints

The API is fully documented with interactive Swagger/OpenAPI UI at `/docs`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Responsive Single-Page Application terminal |
| `GET` | `/health` | Engine health check & computational capabilities metadata |
| `GET` | `/api/v1/market/borrower?ticker=MSFT` | Fills the Merton inputs and annual financials for a listed borrower from public data |
| `GET` | `/api/v1/market/companies` | Every SEC ticker for the dropdowns, flagged when Load can fill it automatically |
| `POST` | `/api/v1/merton/solve` | Solves latent enterprise assets ($V_A, \sigma_A$) with analytical Jacobian |
| `POST` | `/api/v1/forensics/analyze` | Beneish 8-Factor, Altman $Z''$, and CCC working capital analysis |
| `POST` | `/api/v1/regulatory/vasicek` | Basel ASRF conditional distribution & regulatory capital ($K$) |
| `POST` | `/api/v1/regulatory/ecl` | Evaluates IFRS 9 3-stage transitions and computes multi-period ECL |
| `POST` | `/api/v1/credit/comprehensive` | Unified institutional assessment with internal risk band and indicative stance |
| `POST` | `/api/v1/credit/export-memo` | Compiles & streams binary PDF institutional Credit Risk Memorandum |

---

## 5. Quickstart & Local Setup

### Using Python CLI
```bash
# Clone the repository
git clone https://github.com/ivahcomworkspace-droid/aegis-engine.git
cd aegis-engine

# Install dependencies
pip install -r requirements.txt

# Run the test suite (123 tests, all offline)
python -m pytest tests -v

# Enable live data: put your name or company and an email in AEGIS_SEC_USER_AGENT
cp .env.example .env

# Launch the FastAPI server + Web Terminal
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker
```bash
# Build multi-stage production container
docker build -t aegis-credit-engine .

# Run container on port 8000 (with live-data settings from .env)
docker run --env-file .env -p 8000:8000 aegis-credit-engine
```

Open `http://localhost:8000` to interact with the Live Terminal.

### Live market data (fill the inputs from a ticker)

Pick a company from the **Ticker** or **Company** dropdown in **Borrower Parameters** (or type a ticker and press **Load**). Both dropdowns search all 10,405 SEC tickers by ticker or name; the 6,767 marked **(A)** load automatically and are listed first. Choosing one fills both fields and, for (A) companies, loads the data straight away; typing an exact ticker or company name fills the other field. The engine fills the five Merton inputs and the memo's financial statements, and shows each value's source under its slider:

| Input | Source |
|---|---|
| Equity value ($E$) | Latest price × shares outstanding from the latest 10-Q/10-K cover page |
| Equity volatility ($\sigma_E$) | Annualised standard deviation of a year (252 days) of daily log returns |
| Short-term debt (STD) | Latest balance sheet: debt due within a year plus short-term borrowings |
| Long-term debt (LTD) | Latest balance sheet: long-term debt due after a year |
| Risk-free rate ($r$) | US Treasury 1-year par yield, converted to continuous compounding |
| Memo financials | The last two annual reports (10-K), scored with Beneish, Altman $Z''$ and the cash cycle |

- **Sources:** SEC EDGAR XBRL company facts and the US Treasury yield curve are official and free. Prices come from Yahoo Finance's public chart endpoint, which is free but unofficial: it can throttle or change without notice. Each source fails on its own, so a missing price still leaves debt and the rate filled, with a note saying what to enter by hand.
- **SEC contact (required):** the SEC asks automated clients to identify themselves, so set `AEGIS_SEC_USER_AGENT` (for example `"Acme Credit risk@acme.com"`) in the environment, in `.env`, or in your host's settings (on Render: *Environment*). Without it, **Load** explains what to set.
- **Banks and other financials:** for deposit takers (SIC 6021–6036, or deposits above 10% of liabilities), customer deposits, repo funding and short-term borrowings count as short-term debt, following the SVB case study. Beneish, Altman $Z''$ and the cash cycle are calibrated on non-financial companies, so any financial company (SIC 6000–6999: banks, insurers, real estate) is flagged `is_financial_institution`: the models step aside and its risk band rests on distance to default and IFRS 9 stage, on the dashboard and in the memo. Without a reported cost of sales, the cash cycle counts receivable days only.
- **Coverage:** companies that filed a 10-K or 10-Q in the last three quarters: 6,767 tickers from 5,154 companies as of November 2026. [`docs/supported_companies.csv`](docs/supported_companies.csv) lists every SEC ticker with an `A` for these; regenerate it with `python scripts/list_supported_companies.py`. Foreign filers (20-F, IFRS), funds, private companies and other gaps get a clear message; type any value into the boxes beside the sliders, which widen their range to fit (for example `4.2B` or `35%`).
- **Filings without standard tags:** when debt is only in company-specific tags (Ford, GM), liabilities stand in for it: current liabilities as short-term debt and the rest as long-term, KMV's original default point; an unclassified balance sheet (Berkshire) counts all liabilities as long-term. Companies with several share classes (Berkshire, Visa) are valued class by class from the latest filing's cover page, each listed class at its own price; when the filing doesn't say which class trades (Warner Music, most SPACs), every class is valued at the listed price. A preferred share, warrant, unit or note ticker is priced through the company's common stock, and companies with no listed common stock aren't marked (A). A new holding company without two annual reports yet (ExxonMobil Holdings, 2026) takes them from the registrant it replaced (`PREDECESSOR_CIKS`). Each fallback adds a note on screen.
- **Licensed price feed:** add a provider class (`name`, `label`, `async history(client, symbol)`) to `PRICE_PROVIDERS` in `services/market_data.py` and select it with `AEGIS_PRICE_PROVIDER`.
- **Caching:** filings are cached for 12 hours, Treasury yields for 6 hours and prices for 15 minutes in `.cache/market` (`AEGIS_CACHE_DIR` moves it).

### Credit memo

**Export Underwriting Memo** downloads an A4 PDF built from the same engine assessment the dashboard shows: the dashboard renders the response of `/api/v1/credit/comprehensive` and sends that very payload to `/api/v1/credit/export-memo`, so every figure on paper matches the screen (band, stance, IFRS 9 stage, distance to default, PD, ECL, forensic scores, stress PD). The memo covers:

1. **Summary:** risk band, IFRS 9 stage, indicative stance, six key metrics and the key findings.
2. **Inputs and sources:** each Merton input with its source and as-of date (SEC filing, Treasury yield, price feed, sample scenario or entered by hand).
3. **Merton structural model**, **financial statement forensics** (scorecards, Beneish indices, Altman $Z''$ components, working-capital cycle), **Vasicek regulatory capital**, the **IFRS 9 ECL schedule**, and **method and limitations**.

Money is in $ millions throughout and uses the dashboard's formats (`services/formatting.py`; a test runs the dashboard's own formatters against them). Charts are vector (SVG). The memo is set in Lato, which the Docker image installs (`fonts-lato`); elsewhere it falls back to Liberation Sans or DejaVu Sans. For banks and other financial companies the forensic scores read *n/a*, as on the dashboard.

---

## 6. Empirical Case Studies

### A. Silicon Valley Bank (SIVB) Distress Backtest (Q4 2022 Pre-Insolvency)

- **Structural Merton Inversion:** $V_A = \$101{,}836.5\text{M}$, $\sigma_A = 9.69\%$, $\text{DD} = 1.41\sigma$, $\text{PD}_{\text{physical}} = 7.95\%$, Fair Credit Spread $= 557.0\text{ bps}$.
- **Balance Sheet Forensics:** Raw Beneish $M = -2.08$ (below $-1.78$ threshold), Altman $Z'' = 0.23$ (Deep Distress Zone), Asset Quality Index $\text{AQI} = 1.58$.
- **Financial Institution Guardrail:** When flagged with `is_financial_institution=True`, non-applicable non-financial forensic flags are suppressed with domain warning: *"These models were built for non-financial firms."*
- **Regulatory Staging & ECL:** Stage 3 (Credit Impaired), Lifetime ECL allowance of $\$61{,}321\text{M}$ ($61.32\%$ of $\$100{,}000\text{M}$ EAD).
- **Mandate Verdict:** **Band 7 (Impaired / Default Risk)** — Indicative Stance: Immediate Workout.
- **Filing Timing & Sensitivity:** Reflects FY2022 10-K / Q3 2022 10-Q pre-run financial statements (~70 days prior to March 10, 2023 receivership). Stress testing under rapid deposit run debt barriers ($D \in [\$75\text{B}, \$95\text{B}]$) demonstrates catastrophic solvency collapse.
- **Full Case Study:** Refer to [`docs/case_studies/SVB_Q4_2022_distress_backtest.md`](docs/case_studies/SVB_Q4_2022_distress_backtest.md).

### B. Microsoft Corporation (MSFT) Sovereign-Equivalent Benchmark

- **Structural Merton Inversion:** $V_A = \$2{,}425{,}047.1\text{M}$, $\sigma_A = 22.27\%$, $\text{DD} = 20.58\sigma$, $\text{PD}_{\text{physical}} < 0.0001\%$, Fair Credit Spread $= 0.0\text{ bps}$.
- **Balance Sheet Forensics:** Beneish $M = -2.51$ (Clean, well below $-1.78$ threshold), Altman $Z'' = 4.70$ (Safe Zone $> 2.60$).
- **Regulatory Staging & ECL:** Stage 1 (Performing — 12-Month ECL), 12-Month ECL allowance of $\$0.02$ ($0.0000\%$ of $\$100{,}000\text{M}$ EAD).
- **Mandate Verdict:** **Band 1 (Minimal Credit Risk)** — Indicative Stance: Prime Solvency.
- **Full Case Study:** Refer to [`docs/case_studies/MSFT_fortress_benchmark.md`](docs/case_studies/MSFT_fortress_benchmark.md).
