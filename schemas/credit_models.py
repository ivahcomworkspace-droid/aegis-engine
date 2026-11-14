"""
Aegis Perimeter Defense: Pydantic v2 Strict Validation Models.
Enforces double-entry balance sheet identities, volatility bounds, and physical constraints.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator


class MertonInput(BaseModel):
    """
    Observable inputs required for Merton structural credit model inversion.
    """
    equity_value: float = Field(..., gt=0, description="Observable market capitalization / equity value (E)")
    equity_volatility: float = Field(..., gt=0.005, le=5.0, description="Annualized equity volatility (sigma_E), e.g. 0.40")
    short_term_debt: float = Field(..., ge=0, description="Short-term debt / debt due within 1 year")
    long_term_debt: float = Field(..., ge=0, description="Long-term debt principal")
    risk_free_rate: float = Field(0.045, ge=0.0, le=0.25, description="Annualized continuously compounded risk-free rate (r)")
    drift_mu: Optional[float] = Field(default=None, ge=-0.50, le=1.00, description="Physical asset drift mu_A. Defaults to r if None")
    maturity: float = Field(default=1.0, gt=0.0, le=30.0, description="Debt maturity / evaluation horizon T in years")
    loss_given_default: float = Field(default=0.45, ge=0.0, le=1.0, description="Loss Given Default (LGD) [0, 1]")

    @property
    def default_barrier(self) -> float:
        """Default barrier D = Short-Term Debt + 0.5 * Long-Term Debt"""
        return self.short_term_debt + 0.5 * self.long_term_debt


class MertonOutput(BaseModel):
    """
    Inverted latent asset metrics and distance-to-default results.
    """
    asset_value: float = Field(..., description="Inverted latent enterprise asset value (V_A)")
    asset_volatility: float = Field(..., description="Inverted annualized asset volatility (sigma_A)")
    default_barrier: float = Field(..., description="Merton default barrier (D = STD + 0.5*LTD)")
    d1: float = Field(..., description="Black-Scholes-Merton d1")
    d2: float = Field(..., description="Black-Scholes-Merton d2")
    distance_to_default_physical: float = Field(..., description="Physical Distance-to-Default (DD)")
    default_probability_physical: float = Field(..., description="Physical Probability of Default (Phi(-DD))")
    distance_to_default_risk_neutral: float = Field(..., description="Risk-Neutral Distance-to-Default (DD_RN)")
    default_probability_risk_neutral: float = Field(..., description="Risk-Neutral Probability of Default (Phi(-DD_RN))")
    credit_spread_bps: float = Field(..., description="Risk-neutral theoretical credit spread in basis points")
    structural_merton_spread_bps: Optional[float] = Field(default=None, description="Exact structural Merton credit spread in basis points")
    solver_converged: bool = Field(..., description="Whether Powell hybrid solver converged successfully")
    solver_message: str = Field(..., description="Solver exit flag / diagnostics")
    solver_used: Optional[str] = Field(default=None, description="Solver method used ('hybr', 'trf_fallback', 'zero_debt_exact')")
    residual_norm: Optional[float] = Field(default=None, description="L2 norm of dimensionless Merton system residuals")
    solver_time_ms: float = Field(..., description="Wall-clock solver time in milliseconds")
    leverage_ratio: float = Field(..., description="Debt barrier to Asset value ratio (D / V_A)")


class MertonAnalysisRequest(BaseModel):
    equity_market_cap: float = Field(..., gt=0.0, description="Observable market capitalization ($M)")
    equity_volatility: float = Field(..., gt=0.005, le=5.0, description="Annualized historical equity volatility")
    short_term_debt: float = Field(..., ge=0.0, description="Debt due within 12 months ($M)")
    long_term_debt: float = Field(..., ge=0.0, description="Contractual debt maturing >12 months ($M)")
    risk_free_rate: float = Field(0.045, ge=0.0, le=0.25, description="Continuous risk-free sovereign yield")
    asset_drift_mu: float = Field(0.075, ge=-0.50, le=1.00, description="Expected real-world asset growth drift")
    maturity_years: float = Field(1.0, gt=0.0, le=30.0, description="Option maturity / debt horizon (years)")
    recovery_rate: float = Field(0.40, ge=0.0, le=1.0, description="Senior unsecured recovery expectation")


class ForensicAnalysisRequest(BaseModel):
    # Comparative Income Statement & Balance Sheet Line Items
    sales_t0: float = Field(..., gt=0.0)
    sales_t1: float = Field(..., gt=0.0)
    cogs_t0: float = Field(..., ge=0.0)
    cogs_t1: float = Field(..., ge=0.0)
    receivables_t0: float = Field(..., ge=0.0)
    receivables_t1: float = Field(..., ge=0.0)
    current_assets_t0: float = Field(..., gt=0.0)
    current_assets_t1: float = Field(..., gt=0.0)
    ppe_t0: float = Field(..., ge=0.0)
    ppe_t1: float = Field(..., ge=0.0)
    securities_t0: float = Field(0.0, ge=0.0)
    securities_t1: float = Field(0.0, ge=0.0)
    total_assets_t0: float = Field(..., gt=0.0)
    total_assets_t1: float = Field(..., gt=0.0)
    depreciation_t0: float = Field(..., ge=0.0)
    depreciation_t1: float = Field(..., ge=0.0)
    sga_t0: float = Field(..., ge=0.0)
    sga_t1: float = Field(..., ge=0.0)
    net_income_t0: float = Field(...)
    operating_cash_flow_t0: float = Field(...)
    total_debt_t0: float = Field(..., ge=0.0)
    total_debt_t1: float = Field(..., ge=0.0)
    working_capital_t0: float = Field(...)
    retained_earnings_t0: float = Field(...)
    ebit_t0: float = Field(...)
    book_value_equity_t0: float = Field(...)
    total_liabilities_t0: float = Field(..., gt=0.0)
    inventory_t0: float = Field(0.0, ge=0.0)
    inventory_t1: float = Field(0.0, ge=0.0)
    payables_t0: float = Field(0.0, ge=0.0)
    payables_t1: float = Field(0.0, ge=0.0)

    @model_validator(mode='after')
    def validate_double_entry_reconciliation(self):
        implied_equity = self.total_assets_t0 - self.total_liabilities_t0
        tolerance = 0.01 * self.total_assets_t0
        if abs(implied_equity - self.book_value_equity_t0) > tolerance:
            raise ValueError(
                f"Double-Entry Identity Violation: Total Assets ({self.total_assets_t0}) - "
                f"Total Liabilities ({self.total_liabilities_t0}) = {implied_equity:.2f}, "
                f"which does not reconcile with Book Equity ({self.book_value_equity_t0}) "
                f"within 1% tolerance."
            )
        return self


class ForensicInput(BaseModel):
    """
    Institutional balance sheet and financial statement data across period t and t-1.
    Enforces balance sheet sanity checks and identity relations.
    """
    # Financial institution domain guard
    is_financial_institution: bool = Field(default=False, description="Flag indicating borrower is a bank, insurer, or financial institution")
    domain_warning: Optional[str] = Field(default=None, description="Domain limitation notice")

    # Current period (t)
    sales_t: float = Field(..., gt=0, description="Revenue / Sales in period t")
    cogs_t: float = Field(..., ge=0, description="Cost of Goods Sold in period t")
    receivables_t: float = Field(..., ge=0, description="Accounts Receivable in period t")
    current_assets_t: float = Field(..., ge=0, description="Total Current Assets in period t")
    ppe_t: float = Field(..., ge=0, description="Net Property, Plant and Equipment in period t")
    total_assets_t: float = Field(..., gt=0, description="Total Assets in period t")
    depreciation_t: float = Field(..., ge=0, description="Depreciation and Amortization in period t")
    sga_t: float = Field(..., ge=0, description="Selling, General & Administrative expense in period t")
    net_income_t: float = Field(..., description="Net Income in period t")
    cash_flow_operations_t: float = Field(..., description="Operating Cash Flow (CFO) in period t")
    current_liabilities_t: float = Field(..., ge=0, description="Current Liabilities in period t")
    long_term_debt_t: float = Field(..., ge=0, description="Long-Term Debt in period t")
    total_liabilities_t: float = Field(..., ge=0, description="Total Liabilities in period t")
    inventory_t: float = Field(..., ge=0, description="Total Inventory in period t")
    accounts_payable_t: float = Field(..., ge=0, description="Accounts Payable in period t")
    ebit_t: float = Field(..., description="Operating Income / EBIT in period t")
    retained_earnings_t: float = Field(..., description="Retained Earnings in period t")
    book_value_equity_t: float = Field(..., description="Book Value of Equity in period t")
    securities_t: float = Field(default=0.0, ge=0, description="Marketable securities in period t")
    working_capital_t: Optional[float] = Field(default=None, description="Working capital in period t")

    # Prior period (t-1)
    sales_t_minus_1: float = Field(..., gt=0, description="Revenue / Sales in period t-1")
    cogs_t_minus_1: float = Field(..., ge=0, description="Cost of Goods Sold in period t-1")
    receivables_t_minus_1: float = Field(..., ge=0, description="Accounts Receivable in period t-1")
    current_assets_t_minus_1: float = Field(..., ge=0, description="Total Current Assets in period t-1")
    ppe_t_minus_1: float = Field(..., ge=0, description="Net Property, Plant and Equipment in period t-1")
    total_assets_t_minus_1: float = Field(..., gt=0, description="Total Assets in period t-1")
    depreciation_t_minus_1: float = Field(..., ge=0, description="Depreciation and Amortization in period t-1")
    sga_t_minus_1: float = Field(..., ge=0, description="SG&A expense in period t-1")
    long_term_debt_t_minus_1: float = Field(..., ge=0, description="Long-Term Debt in period t-1")
    total_liabilities_t_minus_1: float = Field(..., ge=0, description="Total Liabilities in period t-1")
    inventory_t_minus_1: float = Field(..., ge=0, description="Total Inventory in period t-1")
    accounts_payable_t_minus_1: float = Field(..., ge=0, description="Accounts Payable in period t-1")
    securities_t_minus_1: float = Field(default=0.0, ge=0, description="Marketable securities in period t-1")
    current_liabilities_t_minus_1: Optional[float] = Field(default=None, ge=0, description="Current Liabilities in period t-1")

    @model_validator(mode='after')
    def validate_balance_sheet_identities(self) -> 'ForensicInput':
        if self.working_capital_t is None:
            self.working_capital_t = self.current_assets_t - self.current_liabilities_t

        # 1. Validation check ensuring current_liabilities <= total_liabilities
        if self.current_liabilities_t > self.total_liabilities_t:
            raise ValueError(
                f"Accounting Identity Failure: Current liabilities ({self.current_liabilities_t}) "
                f"exceeds total liabilities ({self.total_liabilities_t})."
            )
        if hasattr(self, "total_liabilities_t_minus_1") and self.total_liabilities_t_minus_1 is not None:
            if hasattr(self, "current_liabilities_t_minus_1") and self.current_liabilities_t_minus_1 is not None:
                if self.current_liabilities_t_minus_1 > self.total_liabilities_t_minus_1:
                    raise ValueError(
                        f"Accounting Identity Failure: Prior period current liabilities ({self.current_liabilities_t_minus_1}) "
                        f"exceeds prior total liabilities ({self.total_liabilities_t_minus_1})."
                    )

        # 2. Validation check ensuring current_assets <= total_assets
        if self.current_assets_t > self.total_assets_t:
            raise ValueError(
                f"Accounting Identity Failure: Current assets ({self.current_assets_t}) "
                f"exceeds total assets ({self.total_assets_t})."
            )
        if self.current_assets_t_minus_1 > self.total_assets_t_minus_1:
            raise ValueError(
                f"Accounting Identity Failure: Prior period current assets ({self.current_assets_t_minus_1}) "
                f"exceeds prior total assets ({self.total_assets_t_minus_1})."
            )

        # Fix S2: Balance sheet accounting identity validator (tolerance <= 0.01 * Assets)
        if (
            self.total_assets_t is not None
            and self.total_liabilities_t is not None
            and self.book_value_equity_t is not None
        ):
            implied_equity = self.total_assets_t - self.total_liabilities_t
            tolerance = 0.01 * self.total_assets_t
            if abs(implied_equity - self.book_value_equity_t) > tolerance:
                raise ValueError(
                    f"Balance-Sheet Identity Violation: Total Assets ({self.total_assets_t}) - "
                    f"Total Liabilities ({self.total_liabilities_t}) = {implied_equity:.2f}, "
                    f"which does not reconcile with Book Equity ({self.book_value_equity_t}) "
                    f"within 1% tolerance."
                )
        return self


class BeneishResult(BaseModel):
    dsri: float = Field(..., description="Days Sales in Receivables Index")
    gmi: float = Field(..., description="Gross Margin Index")
    aqi: float = Field(..., description="Asset Quality Index")
    sgi: float = Field(..., description="Sales Growth Index")
    depi: float = Field(..., description="Depreciation Index")
    sgai: float = Field(..., description="SG&A Expense Index")
    lvgi: float = Field(..., description="Leverage Index")
    tata: float = Field(..., description="Total Accruals to Total Assets")
    m_score: float = Field(..., description="Beneish 8-Factor M-Score")
    m_score_probit_index: float = Field(..., description="Cumulative normal probit index Phi(M)")
    manipulation_probability: float = Field(..., description="Deprecated alias for m_score_probit_index")
    is_manipulator: bool = Field(..., description="True if M-Score > -1.78 (active red flag threshold)")
    red_flags: List[str] = Field(default_factory=list, description="Specific sub-indices indicating forensic anomalies")
    domain_warning: Optional[str] = Field(default=None, description="Domain limitation notice")
    data_quality_flags: List[str] = Field(default_factory=list, description="Explicit warnings for missing or non-computable metrics")


class AltmanResult(BaseModel):
    x1: float = Field(..., description="Working Capital / Total Assets")
    x2: float = Field(..., description="Retained Earnings / Total Assets")
    x3: float = Field(..., description="EBIT / Total Assets")
    x4: float = Field(..., description="Book Value Equity / Total Liabilities")
    z_double_prime: float = Field(..., description="Altman Z''-Score (Services & Emerging Markets)")
    zone: str = Field(..., description="Classification Zone: Safe, Grey, or Distress")
    distress_assessment: str = Field(..., description="Executive interpretation of insolvency likelihood")
    domain_warning: Optional[str] = Field(default=None, description="Domain limitation notice")


class CCCResult(BaseModel):
    dso_t: float = Field(..., description="Days Sales Outstanding (period t)")
    dio_t: float = Field(..., description="Days Inventory Outstanding (period t)")
    dpo_t: float = Field(..., description="Days Payables Outstanding (period t)")
    ccc_t: float = Field(..., description="Cash Conversion Cycle in days (period t)")
    dso_t_minus_1: float = Field(..., description="Days Sales Outstanding (period t-1)")
    dio_t_minus_1: float = Field(..., description="Days Inventory Outstanding (period t-1)")
    dpo_t_minus_1: float = Field(..., description="Days Payables Outstanding (period t-1)")
    ccc_t_minus_1: float = Field(..., description="Cash Conversion Cycle in days (period t-1)")
    delta_ccc: float = Field(..., description="Change in Cash Conversion Cycle (days)")
    delta_gross_margin: float = Field(..., description="Change in Gross Profit Margin (percentage points)")
    warning_triggered: bool = Field(..., description="Alert if Delta CCC > +20 days while Gross Margin is static")
    alert_message: Optional[str] = Field(default=None, description="Actionable working capital stress diagnostic")
    domain_warning: Optional[str] = Field(default=None, description="Domain limitation notice")


class ForensicAnalysisResult(BaseModel):
    beneish: BeneishResult
    altman: AltmanResult
    ccc: CCCResult
    composite_forensic_risk: str = Field(..., description="Composite risk tier: Low, Medium, High, or Critical")
    summary_findings: List[str] = Field(default_factory=list, description="Forensic credit investigator bullet findings")
    is_financial_institution: bool = Field(default=False, description="Whether entity is a financial institution")
    domain_warning: Optional[str] = Field(default=None, description="Domain limitation notice")


class VasicekInput(BaseModel):
    """
    Vasicek Asymptotic Single Risk Factor (ASRF) macro credit loss distribution parameters.
    """
    pit_pd: float = Field(..., gt=0.0, lt=1.0, description="Point-in-Time Probability of Default (PD_PIT)")
    macro_shock_z: Optional[float] = Field(
        default=None,
        description="Systemic standard normal factor shock Z. If None, computes adverse Basel 99.9% VaR quantile"
    )
    confidence_level: float = Field(default=0.999, gt=0.5, lt=1.0, description="Regulatory capital quantile (e.g. 0.999)")
    lgd: float = Field(default=0.45, ge=0.0, le=1.0, description="Downturn Loss Given Default")
    ead: float = Field(default=1_000_000.0, gt=0.0, description="Exposure at Default in currency units")
    maturity: float = Field(default=1.0, ge=1.0, description="Effective loan maturity T (years)")
    firm_sales_eur_millions: Optional[float] = Field(
        default=None,
        description="Annual sales in EUR millions for Basel SME asset correlation adjustment (5m - 50m)"
    )


class VasicekOutput(BaseModel):
    pit_pd: float
    asset_correlation_rho: float = Field(..., description="ASRF corporate asset correlation rho in [0.12, 0.24]")
    conditional_pd: float = Field(..., description="Conditional PD under specified macro shock / quantile")
    regulatory_capital_k: float = Field(..., description="Basel IRB regulatory capital requirement ratio K")
    unexpected_loss: float = Field(..., description="Unexpected Loss (UL) in currency units")
    expected_loss: float = Field(..., description="Expected Loss (EL = PD * LGD * EAD) in currency units")
    economic_capital_ead: float = Field(..., description="Total capital buffer required (UL + EL)")
    stress_scenarios: Dict[str, float] = Field(..., description="Conditional PD across standard macro stress tiers")


class IFRS9StageInput(BaseModel):
    current_distance_to_default: float = Field(..., description="Current evaluation Distance-to-Default (DD_t)")
    origination_distance_to_default: float = Field(..., description="Inception / origination Distance-to-Default (DD_0)")
    current_pd: float = Field(..., gt=0.0, le=1.0, description="Current 1-year annualized default probability")
    days_past_due: int = Field(default=0, ge=0, description="Days Past Due (DPD) on debt service obligations")
    is_credit_impaired: bool = Field(default=False, description="Manual qualitative or legal impairment flag")
    ead_schedule: List[float] = Field(..., min_length=1, description="EAD principal balance amortization schedule")
    effective_interest_rate: float = Field(default=0.06, ge=0.0, description="Contractual Effective Interest Rate (EIR)")
    lgd_schedule: Optional[List[float]] = Field(default=None, description="Per-period LGD schedule (default 0.45)")


class ECLPeriodSchedule(BaseModel):
    period: int
    marginal_pd: float
    cumulative_pd: float
    survival_probability: float
    ead: float
    lgd: float
    discount_factor: float
    discounted_ecl: float


class ECLResult(BaseModel):
    assigned_stage: int = Field(..., description="IFRS 9 Stage: 1 (Performing), 2 (Underperforming), or 3 (Impaired)")
    stage_name: str = Field(..., description="Stage 1 (12M ECL), Stage 2 (Lifetime ECL), or Stage 3 (Impaired ECL)")
    stage_rationale: str = Field(..., description="Regulatory justification for stage classification")
    delta_distance_to_default: float = Field(..., description="Change in Distance-to-Default (DD_current - DD_origination)")
    total_ecl: float = Field(..., description="Regulatory Expected Credit Loss allowance amount")
    is_lifetime_ecl: bool = Field(..., description="True if lifetime ECL provisioned (Stage 2 or 3)")
    ecl_percentage_of_ead: float = Field(..., description="ECL allowance divided by initial EAD (%)")
    period_schedules: List[ECLPeriodSchedule] = Field(..., description="Multi-period cash flow and loss schedule")


class InputSource(BaseModel):
    """Where one input came from, shown in the memo's inputs table."""
    source: str = Field(..., max_length=240, description="Data source, e.g. 'SEC 10-Q balance sheet'")
    as_of: Optional[str] = Field(default=None, max_length=40, description="Date the value refers to")


class ComprehensiveCreditRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    company_name: str = Field(...)
    
    # Merton Inputs
    equity_market_cap: float = Field(..., gt=0.0, description="Observable market capitalization ($M)")
    equity_volatility: float = Field(..., gt=0.005, le=5.0, description="Annualized equity volatility")
    short_term_debt: float = Field(..., ge=0.0, description="Debt due within 12 months ($M)")
    long_term_debt: float = Field(..., ge=0.0, description="Debt maturing >12 months ($M)")
    risk_free_rate: float = Field(0.045, ge=0.0, le=0.25, description="Continuous risk-free sovereign yield")
    asset_drift_mu: float = Field(0.075, ge=-0.50, le=1.00, description="Expected real-world asset growth drift")
    loan_exposure: float = Field(100.0, gt=0.0, description="Total loan or facility exposure amount ($M)")
    recovery_rate: float = Field(0.40, ge=0.0, le=1.0, description="Senior unsecured recovery expectation")
    maturity_years: float = Field(1.0, gt=0.0, le=10.0, description="Option maturity / debt horizon (years)")

    # Simplified Balance Sheet for In-Flight Underwriting
    total_assets: float = Field(..., gt=0.0, description="Total balance sheet enterprise assets ($M)")
    total_liabilities: float = Field(..., gt=0.0, description="Total liabilities ($M)")
    book_value_equity: float = Field(..., description="Reported book value of equity ($M)")
    working_capital: float = Field(..., description="Net working capital ($M)")
    retained_earnings: float = Field(..., description="Accumulated retained earnings ($M)")
    ebit: float = Field(..., description="Earnings before interest and taxes ($M)")

    # Working Capital Ratios
    receivables: float = Field(50.0, ge=0.0)
    inventory: float = Field(60.0, ge=0.0)
    payables: float = Field(45.0, ge=0.0)
    revenue: float = Field(350.0, gt=0.0)
    cogs: float = Field(240.0, ge=0.0)

    # Forensic & Stress Simulation Flags
    simulate_beneish_manipulation: bool = Field(False)
    simulate_working_capital_stress: bool = Field(False)
    is_financial_institution: bool = Field(default=False, description="Flag indicating borrower is a bank, insurer, or financial institution")

    # Optional nested wrappers for backward compatibility
    merton_input: Optional[MertonInput] = Field(default=None)
    forensic_input: Optional[ForensicInput] = Field(default=None)
    origination_distance_to_default: Optional[float] = Field(default=None)
    days_past_due: int = Field(default=0, ge=0)
    effective_interest_rate: float = Field(default=0.06, ge=0.0)
    loan_tenor_years: int = Field(default=5, ge=1, le=30)

    # Provenance for the memo: where each Merton input came from and which statements fed the forensics
    input_sources: Dict[str, InputSource] = Field(
        default_factory=dict,
        description="Source of each input, keyed by equity_value, equity_volatility, short_term_debt, "
                    "long_term_debt and risk_free_rate",
    )
    forensic_basis: Optional[str] = Field(
        default=None, max_length=200,
        description="Statements behind the forensic scores, e.g. 'Filed financials · FY2025 vs FY2024'",
    )

    @model_validator(mode='before')
    @classmethod
    def populate_from_nested(cls, data: Any) -> Any:
        if isinstance(data, dict):
            def _extract_val(val: Any) -> Any:
                if isinstance(val, dict):
                    return val.get("value")
                return val

            m_in = data.get("merton_input")
            f_in = data.get("forensic_input")
            if m_in and isinstance(m_in, dict):
                eq_v = _extract_val(m_in.get("equity_value"))
                if eq_v is not None:
                    data.setdefault("equity_market_cap", float(eq_v))
                vol_v = _extract_val(m_in.get("equity_volatility"))
                if vol_v is not None:
                    data.setdefault("equity_volatility", float(vol_v))
                std_v = _extract_val(m_in.get("short_term_debt"))
                if std_v is not None:
                    data.setdefault("short_term_debt", float(std_v))
                ltd_v = _extract_val(m_in.get("long_term_debt"))
                if ltd_v is not None:
                    data.setdefault("long_term_debt", float(ltd_v))
                r_v = _extract_val(m_in.get("risk_free_rate"))
                if r_v is not None:
                    data.setdefault("risk_free_rate", float(r_v))
                mat_v = _extract_val(m_in.get("maturity"))
                if mat_v is not None:
                    data.setdefault("maturity_years", float(mat_v))
                lgd_v = _extract_val(m_in.get("loss_given_default"))
                if lgd_v is not None:
                    data.setdefault("recovery_rate", 1.0 - float(lgd_v))
                std_num = float(_extract_val(m_in.get("short_term_debt")) or 0.0)
                ltd_num = float(_extract_val(m_in.get("long_term_debt")) or 0.0)
                if std_num + ltd_num > 0:
                    data.setdefault("loan_exposure", std_num + ltd_num)

            # Direct root-level aliases and overrides from active frontend state
            if "equity_value" in data and "equity_market_cap" not in data:
                eq_v = _extract_val(data["equity_value"])
                if eq_v is not None:
                    data["equity_market_cap"] = float(eq_v)
            if "equity_vol" in data and "equity_volatility" not in data:
                vol_v = _extract_val(data["equity_vol"])
                if vol_v is not None:
                    data["equity_volatility"] = float(vol_v)
            if "rfr" in data and "risk_free_rate" not in data:
                r_v = _extract_val(data["rfr"])
                if r_v is not None:
                    data["risk_free_rate"] = float(r_v)
            if "simulate_manipulation" in data and "simulate_beneish_manipulation" not in data:
                data["simulate_beneish_manipulation"] = bool(data["simulate_manipulation"])
            if "simulate_ccc_stress" in data and "simulate_working_capital_stress" not in data:
                data["simulate_working_capital_stress"] = bool(data["simulate_ccc_stress"])
            if "chkManipulator" in data and "simulate_beneish_manipulation" not in data:
                data["simulate_beneish_manipulation"] = bool(data["chkManipulator"])
            if "chkCCCAlert" in data and "simulate_working_capital_stress" not in data:
                data["simulate_working_capital_stress"] = bool(data["chkCCCAlert"])

            if f_in and isinstance(f_in, dict):
                ta = _extract_val(f_in.get("total_assets_t"))
                if ta is not None:
                    data.setdefault("total_assets", float(ta))
                tl = _extract_val(f_in.get("total_liabilities_t"))
                if tl is not None:
                    data.setdefault("total_liabilities", float(tl))
                bve = _extract_val(f_in.get("book_value_equity_t"))
                if bve is not None:
                    data.setdefault("book_value_equity", float(bve))
                if "is_financial_institution" in f_in:
                    data.setdefault("is_financial_institution", bool(f_in["is_financial_institution"]))
                wc = _extract_val(f_in.get("working_capital_t"))
                if wc is None and "current_assets_t" in f_in and "current_liabilities_t" in f_in:
                    ca = _extract_val(f_in.get("current_assets_t"))
                    cl = _extract_val(f_in.get("current_liabilities_t"))
                    if ca is not None and cl is not None:
                        wc = float(ca) - float(cl)
                if wc is not None:
                    data.setdefault("working_capital", float(wc))
                re = _extract_val(f_in.get("retained_earnings_t"))
                if re is not None:
                    data.setdefault("retained_earnings", float(re))
                ebit = _extract_val(f_in.get("ebit_t"))
                if ebit is not None:
                    data.setdefault("ebit", float(ebit))
                rec = _extract_val(f_in.get("receivables_t"))
                if rec is not None:
                    data.setdefault("receivables", float(rec))
                inv = _extract_val(f_in.get("inventory_t"))
                if inv is not None:
                    data.setdefault("inventory", float(inv))
                pay = _extract_val(f_in.get("accounts_payable_t"))
                if pay is not None:
                    data.setdefault("payables", float(pay))
                rev = _extract_val(f_in.get("sales_t"))
                if rev is not None:
                    data.setdefault("revenue", float(rev))
                cogs = _extract_val(f_in.get("cogs_t"))
                if cogs is not None:
                    data.setdefault("cogs", float(cogs))
        return data

    @model_validator(mode='after')
    def validate_balance_sheet(self):
        implied_equity = self.total_assets - self.total_liabilities
        tolerance = 0.01 * self.total_assets
        if abs(implied_equity - self.book_value_equity) > tolerance:
            raise ValueError(
                f"Accounting Identity Failure: Assets ({self.total_assets}) - "
                f"Liabilities ({self.total_liabilities}) != Book Equity ({self.book_value_equity}) "
                f"within 1% tolerance."
            )
        return self


class AegisInternalRiskBand(str, Enum):
    BAND_1 = "Band 1 (Minimal Credit Risk)"
    BAND_2 = "Band 2 (Low Credit Risk)"
    BAND_3 = "Band 3 (Moderate Credit Risk)"
    BAND_4 = "Band 4 (Acceptable / Monitored)"
    BAND_5 = "Band 5 (Special Mention / Elevated)"
    BAND_6 = "Band 6 (High Risk / Substandard)"
    BAND_7 = "Band 7 (Impaired / Default Risk)"


DEFAULT_REGULATORY_DISCLAIMER = (
    "Aegis internal risk bands and default metrics are uncalibrated quantitative model outputs "
    "derived from mathematical simplifications. They do not constitute credit ratings from an NRSRO "
    "or formalized credit committee approvals."
)


class ComprehensiveCreditResponse(BaseModel):
    ticker: Optional[str] = Field(default=None, description="Borrower equity ticker / identifier")
    company_name: str = Field(..., description="Legal corporate borrower name")
    internal_risk_band: str = Field(
        ...,
        description="Aegis model-implied proprietary risk band (Band 1 through Band 7)"
    )
    indicative_stance: str = Field(
        ...,
        description="Non-binding underwriting posture (e.g., INDICATIVE: LOW RISK / MONITOR)"
    )
    # Retained for backwards-compatibility; mapped directly to internal_risk_band
    credit_rating: str = Field(
        ...,
        description="Internal model risk band (formerly credit rating)"
    )
    recommendation: str = Field(
        ...,
        description="Indicative stance (formerly binary recommendation)"
    )
    regulatory_disclaimer: str = Field(
        default=DEFAULT_REGULATORY_DISCLAIMER,
        description="Model output screening disclaimer (not a credit rating or investment advice)"
    )
    merton: Union[MertonOutput, Dict[str, Any]]
    forensics: Union[ForensicAnalysisResult, Dict[str, Any]]
    regulatory_capital: Union[VasicekOutput, Dict[str, Any]]
    ifrs9_ecl: Union[ECLResult, Dict[str, Any]]

    # Backwards compatibility fields for legacy clients, tests, and memo generator
    composite_rating: Optional[str] = Field(
        default=None,
        description="Legacy alias for internal_risk_band"
    )
    vasicek: Optional[Union[VasicekOutput, Dict[str, Any]]] = Field(
        default=None,
        description="Legacy alias for regulatory_capital"
    )
    inputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inputs the assessment used ($M for money, decimals for rates)"
    )
    input_sources: Dict[str, InputSource] = Field(default_factory=dict, description="Source of each input")
    forensic_basis: Optional[str] = Field(default=None, description="Statements behind the forensic scores")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        description="Evaluation UTC timestamp"
    )
    executive_summary: Optional[str] = Field(
        default=None,
        description="Quantitative risk appraisal narrative"
    )

    @model_validator(mode='after')
    def sync_backwards_compatibility(self) -> 'ComprehensiveCreditResponse':
        if not self.credit_rating:
            self.credit_rating = self.internal_risk_band
        if not self.composite_rating:
            self.composite_rating = self.internal_risk_band
        if not self.recommendation:
            self.recommendation = self.indicative_stance
        if self.vasicek is None and self.regulatory_capital is not None:
            self.vasicek = self.regulatory_capital
        elif self.regulatory_capital is None and self.vasicek is not None:
            self.regulatory_capital = self.vasicek
        return self


# Backwards compatibility alias
ComprehensiveCreditAssessment = ComprehensiveCreditResponse


class MarketDataPoint(BaseModel):
    """One auto-filled input with where it came from."""
    value: Optional[float] = Field(default=None, description="Filled value ($M for money, decimal for rates); null if unavailable")
    source: str = Field(..., description="Data source")
    as_of: Optional[str] = Field(default=None, description="Date the value refers to (YYYY-MM-DD)")
    detail: Optional[str] = Field(default=None, description="How the value was built")


class BorrowerMarketData(BaseModel):
    """Merton inputs and annual financials for a listed borrower, built from SEC filings,
    US Treasury yields and market prices."""
    ticker: str
    company_name: str
    sec_name: Optional[str] = None
    cik: str
    sic: Optional[str] = None
    sic_description: Optional[str] = None
    is_bank: bool = Field(..., description="Deposit taker: deposits count as short-term debt")
    equity_value: MarketDataPoint
    equity_volatility: MarketDataPoint
    short_term_debt: MarketDataPoint
    long_term_debt: MarketDataPoint
    risk_free_rate: MarketDataPoint
    forensic_input: Optional[ForensicInput] = Field(default=None, description="Last two annual reports, $M")
    forensic_analysis: Optional[ForensicAnalysisResult] = None
    forensics_applicable: bool = Field(
        default=True,
        description="False for financial companies (SIC 6000-6999); their forensic_input sets is_financial_institution",
    )
    fiscal_year_end: Optional[str] = None
    prior_fiscal_year_end: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    loaded_at: str
