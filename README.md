# Aegis // Institutional Structural Credit Risk & Forensics Engine

[![Live Web Terminal](https://img.shields.io/badge/Live%20Demo-Render-00c7b7?style=for-the-badge&logo=render&logoColor=white)](https://aegis-engine-s96v.onrender.com)
[![OpenAPI Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://aegis-engine-s96v.onrender.com/docs)
[![Pytest Suite](https://img.shields.io/badge/Tests-57%20Passed-brightgreen?style=for-the-badge)](tests/test_engine.py)

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

   Where corporate asset correlation $\rho$ follows the Basel corporate specification:

$$
\rho = 0.12 \left(\frac{1 - e^{-50 \cdot \text{PD}}}{1 - e^{-50}}\right) + 0.24 \left(1 - \frac{1 - e^{-50 \cdot \text{PD}}}{1 - e^{-50}}\right), \quad \rho \in [0.12, 0.24]
$$

2. **Multi-Period Lifetime Expected Credit Loss (ECL):**

$$
\text{ECL} = \sum_{t=1}^T \frac{\text{PD}_t^* \times \text{LGD}_t \times \text{EAD}_t}{(1 + \text{EIR})^t}
$$

   - **Stage 1 (Performing):** 12-Month ECL ($T = 1$).
   - **Stage 2 (Underperforming / SICR):** Lifetime ECL triggered when $\Delta\text{DD} < -1.0\sigma$ or $\text{DPD} \in [30, 89]$.
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
│   └── memo_generator.py     # Matplotlib visualizer & WeasyPrint/ReportLab PDF compiler
├── api/
│   ├── __init__.py
│   └── main.py               # Asynchronous FastAPI microservice & REST endpoints
├── frontend/
│   └── index.html            # Responsive glassmorphic dark-theme Single-Page App
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest root configuration
│   └── test_engine.py        # 57 deterministic unit, numerical & integration tests
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
| `POST` | `/api/v1/merton/solve` | Solves latent enterprise assets ($V_A, \sigma_A$) with analytical Jacobian |
| `POST` | `/api/v1/forensics/analyze` | Beneish 8-Factor, Altman $Z''$, and CCC working capital analysis |
| `POST` | `/api/v1/regulatory/vasicek` | Basel ASRF conditional distribution & regulatory capital ($K$) |
| `POST` | `/api/v1/regulatory/ecl` | Evaluates IFRS 9 3-stage transitions and computes multi-period ECL |
| `POST` | `/api/v1/credit/comprehensive` | Unified institutional assessment with rating and committee directive |
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

# Run the test suite (57 tests)
python -m pytest tests/test_engine.py -v

# Launch the FastAPI server + Web Terminal
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker
```bash
# Build multi-stage production container
docker build -t aegis-credit-engine .

# Run container on port 8000
docker run -p 8000:8000 aegis-credit-engine
```

Open `http://localhost:8000` to interact with the Live Terminal.

---

## 6. Empirical Case Studies

### Silicon Valley Bank (SIVB) Distress Backtest

> Under review: Beneish result and related figures are being recomputed after a coefficient correction. Do not rely on the M-score reported here.

Refer to [`docs/case_studies/SVB_Q4_2022_distress_backtest.md`](docs/case_studies/SVB_Q4_2022_distress_backtest.md) for the detailed historical pre-insolvency audit.
