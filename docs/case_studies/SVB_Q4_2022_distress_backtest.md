> **Disclaimer:** Model output for screening purposes only. Not a credit rating, credit opinion or investment advice.

# Empirical Backtest: Silicon Valley Bank (SIVB) — Q4 2022 Pre-Insolvency Forensic Audit

**Target Entity:** SVB Financial Group (NASDAQ: SIVB)  
**Evaluation Horizon:** Q4 2022 (As of December 31, 2022 disclosures, ~70 days prior to FDIC receivership on March 10, 2023)  
**Quantitative Mandate:** Institutional Structural Credit Inversion, Forensic Accruals Audit, and IFRS 9 Multi-Period Staging  
**Final Engine Verdict:** **BAND 7 (IMPAIRED / DEFAULT RISK) / STAGE 3 WORKOUT**

---

## 1. Executive Summary & Regulatory Context

In Q4 2022, Silicon Valley Bank maintained investment-grade credit ratings from major legacy credit rating agencies (**Moody's: A1**, **S&P: A**). The rating agencies anchored their credit appraisals to traditional accounting capital metrics (CET1 ratio $> 12\%$, Total Risk-Based Capital ratio $> 15\%$), which completely obscured off-balance-sheet unhedged duration losses in the bank's Held-to-Maturity (HTM) fixed-income portfolio.

By running observable market and regulatory disclosures through the **Aegis Structural Credit Risk & Forensics Engine**, an underwriting committee would have identified catastrophic solvency deterioration **months prior to regulatory collapse**:

1. **Structural Default Barrier Penetration:** Physical Distance-to-Default ($\text{DD}$) collapsed to **$1.41\sigma$** (Physical $\text{PD} = 7.95\%$), penetrating the severe structural distress boundary ($< 1.50\sigma$).
2. **Fair Credit Spread Escalation:** Risk-neutral fair credit spread exploded past **$557.0\text{ bps}$**, while secondary cash bonds traded near par ($< 150\text{ bps}$ spread), signaling an unprecedented debt mispricing arbitrage.
3. **Forensic Accounting Diagnostics & Bank Domain Boundary:**
   - Raw Beneish M-Score with corrected 1999 coefficients yields **$-2.08$** (below the $-1.78$ screening threshold), while Altman $Z''$-Score plummeted into the severe distress zone at **$0.23$** ($< 1.10$).
   - Because standard forensic models (Beneish, Altman $Z''$, CCC) were calibrated on mercantile, industrial, and service firms, evaluating a regulated bank triggers the institutional guard `is_financial_institution=True` with explicit domain warning: *"These models were built for non-financial firms."*
4. **Mandatory IFRS 9 Stage 3 Staging:** Stressed liquidity and credit impairment forced an immediate transition to **Stage 3 (Credit Impaired)** with lifetime loss provisioning of **$\$61,321\text{M}$** (**$61.32\%$ of Initial Exposure at Default**).

---

## 2. Historical Balance Sheet & Market Pricing Inputs

The backtest uses audited financial figures extracted from SVB Financial Group's Q3/Q4 2022 SEC filings (Form 10-Q and 10-K) and market-traded equity options data:

| Parameter | Notation | Reported Value | Economic Interpretation |
| :--- | :--- | :--- | :--- |
| **Observable Market Capitalization** | $E$ | **$\$13,500\text{M}$** | Collapsed from peak of $\$44,000\text{M}$ (~$70\%$ drawdown) |
| **Annualized Equity Volatility** | $\sigma_E$ | **$68.0\%$** | Realized/implied volatility of secondary equity shares |
| **Short-Term Obligations / At-Risk Deposits** | $\text{STD}$ | **$\$85,000\text{M}$** | Uninsured venture demand deposits subject to run risk |
| **Long-Term Contractual Debt** | $\text{LTD}$ | **$\$15,000\text{M}$** | Senior notes, subordinated debt, and FHLB advances |
| **Merton Default Barrier** | $D$ | **$\$92,500\text{M}$** | $D = \text{STD} + 0.5 \times \text{LTD}$ (KMV convention) |
| **Risk-Free Sovereign Rate** | $r$ | **$4.25\%$** | 1-Year US Treasury benchmark yield (Dec 2022) |
| **Real Asset Growth Drift** | $\mu$ | **$4.50\%$** | Projected real asset growth ($\mu \ge r = 4.25\%$) |
| **Maturity Horizon** | $T$ | **$1.0\text{ Year}$** | Standard institutional credit appraisal window |
| **Senior Unsecured Recovery Rate** | $R$ | **$35.0\%$** | Distressed bank resolution recovery assumption ($\text{LGD} = 65\%$) |

---

## 3. Structural Credit Inversion Kernel (Merton Bivariate System)

### Mathematical Formulation
The Aegis Merton engine inverts unobservable latent firm asset value $V_A$ and instantaneous asset volatility $\sigma_A$ by solving the non-linear coupled system with analytical Jacobian:

$$F_1(V_A, \sigma_A) = V_A \Phi(d_1) - D e^{-rT} \Phi(d_2) - E = 0$$

$$F_2(V_A, \sigma_A) = \left(\frac{V_A}{E}\right) \Phi(d_1) \sigma_A - \sigma_E = 0$$

where:
$$d_1 = \frac{\ln(V_A / D) + (r + 0.5\sigma_A^2)T}{\sigma_A \sqrt{T}}, \quad d_2 = d_1 - \sigma_A \sqrt{T}$$

### Numerical Solution Outputs
Supplying the analytical Jacobian matrix $\mathbf{J}$ yields rapid convergence:

| Metric | Analytical Output | Benchmark / Normal Range | Diagnostic Status |
| :--- | :--- | :--- | :--- |
| **Latent Enterprise Assets ($V_A$)** | **$\$101,836.5\text{M}$** | $\$212,000\text{M}$ (Book Assets) | **Market value of assets $< 50\%$ of book** |
| **Implied Asset Volatility ($\sigma_A$)** | **$9.69\%$** | $2.0\% - 4.0\%$ (Commercial Banks) | **$2.5\times$ historical commercial bank baseline** |
| **Physical Distance-to-Default ($\text{DD}$)** | **$1.41\sigma$** | $> 3.50\sigma$ (Investment Grade) | **Severe Distress Boundary ($< 1.50\sigma$)** |
| **Physical 1-Year Default Probability** | **$7.95\%$** | $< 0.05\%$ (Moody's A-equivalent) | **$160\times$ higher than agency-implied PD** |
| **Risk-Neutral 1-Year Default Probability** | **$8.33\%$** | $< 0.10\%$ | Capital buffer depletion threshold |
| **Fair Implied Credit Spread** | **$557.0\text{ bps}$** | $120 - 160\text{ bps}$ (Market Traded) | **$+400\text{ bps}$ blowout** |
| **Structural Merton Spread** | **$557.0\text{ bps}$** | $120 - 160\text{ bps}$ | **Severe risk premium pricing** |
| **Structural Market Leverage ($D / V_A$)** | **$90.83\%$** | $< 70.0\%$ | Extreme insolvency gearing |

$$\text{DD}_{\text{physical}} = \frac{\ln(101,836.5 / 92,500) + (0.045 - 0.5 \times 0.0969^2) \times 1.0}{0.0969 \times 1.0} \approx 1.41\sigma$$

### Filing Timing Caveat & Debt-Barrier Sensitivity
- **Filing Timing Caveat:** SVB's financial statements were compiled as of Q4 2022 (~70 days prior to the March 10, 2023 receivership). Because bank runs accelerate non-linearly, a robust underwriting workflow conducts sensitivity analysis across short-term deposit withdrawal assumptions:

| Short-Term Deposits ($\text{STD}$) | Default Barrier ($D$) | Latent Assets ($V_A$) | Asset Vol ($\sigma_A$) | Distance-to-Default ($\text{DD}$) | Physical PD | Fair Credit Spread |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\$75,000\text{M}$** | $\$82,500\text{M}$ | $\$92,267.8\text{M}$ | $10.66\%$ | **$1.42\sigma$** | $7.80\%$ | $544.1\text{ bps}$ |
| **$\$85,000\text{M}$** (Baseline) | $\$92,500\text{M}$ | $\$101,836.5\text{M}$ | $9.69\%$ | **$1.41\sigma$** | $7.95\%$ | $557.0\text{ bps}$ |
| **$\$95,000\text{M}$** | $\$102,500\text{M}$ | $\$111,407.4\text{M}$ | $8.88\%$ | **$1.40\sigma$** | $8.06\%$ | $567.7\text{ bps}$ |

Across all deposit scenarios ($75B to $95B), Distance-to-Default remains pinned below $1.50\sigma$ and credit spreads exceed $540\text{ bps}$, confirming acute insolvency regardless of deposit runoff pacing.

---

## 4. Forensic Balance Sheet Audit (Beneish & Altman Z'')

### Beneish 8-Factor Model (Beneish 1999 Standard)

$$\text{M} = -4.84 + 0.920 \cdot \text{DSRI} + 0.528 \cdot \text{GMI} + 0.404 \cdot \text{AQI} + 0.892 \cdot \text{SGI} + 0.115 \cdot \text{DEPI} - 0.172 \cdot \text{SGAI} + 4.679 \cdot \text{TATA} - 0.327 \cdot \text{LVGI}$$

```
Beneish M-Score Formula Factors for SVB Q4 2022:
- DSRI (Days Sales in Receivables Index):  1.00  (Normal)
- GMI  (Gross Margin Index):              1.01  (Normal)
- AQI  (Asset Quality Index):             1.58  (CRITICAL ALERT > 1.10)
- SGI  (Sales Growth Index):              1.14  (Normal)
- DEPI (Depreciation Index):              1.00  (Normal)
- SGAI (SG&A Overhead Index):             0.94  (Normal)
- LVGI (Leverage Index):                  1.14  (Normal)
- TATA (Total Accruals to Assets):        0.016 (Accrual disconnect)
---------------------------------------------------------------------------------
RAW BENEISH M-SCORE:                      -2.08  (THRESHOLD: -1.78)
FLAG STATUS:                              CLEAN / BELOW THRESHOLD
```

### Commercial Bank Structural Domain Limitations (`is_financial_institution=True`)
Beneish M-Score and Altman $Z''$ were developed and calibrated for non-financial commercial and manufacturing enterprises. Commercial banks have structural balance-sheet characteristics that limit non-financial ratios:
1. **Deposits are Operating Liabilities:** Demand and time deposits represent the bank's core funding, not commercial trade credit. Rapid leverage shifts reflect deposit flight or liquidity hoarding rather than traditional vendor stretch.
2. **Assets are Financial Claims:** Banks hold loans and marketable debt securities (HTM/AFS) rather than inventory or physical PP&E. Consequently, traditional Working Capital ($X_1$) and Asset Quality Index ($\text{AQI}$) capture portfolio mark-to-market shifts rather than deferred cost capitalization.
3. **Explicit Domain Warning:** When evaluated with `is_financial_institution=True`, Aegis automatically suppresses false-positive non-financial flags and reports:  
   `"These models were built for non-financial firms."`

### Altman $Z''$-Score (Services & Emerging Markets Calibration)
$$Z'' = 6.56 X_1 + 3.26 X_2 + 6.72 X_3 + 1.05 X_4 = \mathbf{0.23}$$

- **Classification Zone:** **DISTRESS ZONE ($Z'' < 1.10$)**  
- Confirms massive capital depletion and working capital inversion.

---

## 5. Regulatory IFRS 9 Multi-Period ECL Staging

Under IFRS 9 / CECL standards, facilities transition from 12-month ECL to Lifetime ECL upon a Significant Increase in Credit Risk ($\text{SICR}$) and to **Stage 3 (Credit Impaired)** upon default or objective impairment:

- **Assigned Stage:** **Stage 3 (Credit Impaired)**
- **Lifetime Allowance:** **$\$61,321\text{M}$**
- **ECL Provisioning % of Initial EAD:** **$61.32\%$**

---

## 6. Underwriting Directive & Executive Summary

```
========================================================================================
                      AEGIS CREDIT RISK UNDERWRITING DIRECTIVE
========================================================================================
BORROWER ENTITY:           SVB FINANCIAL GROUP (NASDAQ: SIVB)
EVALUATION DATE:           DECEMBER 31, 2022 (Q4 DISCLOSURES)
INTERNAL RISK BAND:        BAND 7 (IMPAIRED / DEFAULT RISK)
INDICATIVE STANCE:         INDICATIVE: VERY HIGH RISK / WORKOUT REVIEW
IFRS 9 PROVISIONING:       STAGE 3 LIFETIME PROVISION ($61,321M / 61.32% EAD)
========================================================================================
```

### Forensic Investigator Summary Bulletins:
1. **Unhedged Duration Mismatch:** HTM portfolio market-value losses exceeded $\$15\text{B}$, fully wiping out reported Tangible Common Equity ($\$11.8\text{B}$).
2. **Deposit Run Velocity:** Short-term deposit obligations ($\$85\text{B}$) concentrated in venture capital tech clients with high withdrawal correlation.
3. **Reproducibility:** Run `py scripts/run_case_studies.py` to regenerate all empirical metrics.
