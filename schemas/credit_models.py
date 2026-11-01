"""
Aegis Credit Models & Pydantic v2 Schemas
Institutional balance sheet, structural Merton, forensics, and IFRS 9/CECL data contracts.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class MertonInput(BaseModel):
    """
    Observable inputs required for Merton structural credit model inversion.
    """
    equity_value: float = Field(..., gt=0, description="Observable market capitalization / equity value (E)")
    equity_volatility: float = Field(..., gt=0, description="Annualized equity volatility (sigma_E), e.g. 0.40")
    short_term_debt: float = Field(..., ge=0, description="Short-term debt / debt due within 1 year")
    long_term_debt: float = Field(..., ge=0, description="Long-term debt principal")
    risk_free_rate: float = Field(..., description="Annualized continuously compounded risk-free rate (r), e.g. 0.045")
    drift_mu: Optional[float] = Field(default=None, description="Physical asset drift mu_A. Defaults to r if None")
    maturity: float = Field(default=1.0, gt=0, description="Debt maturity / evaluation horizon T in years")
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
    solver_converged: bool = Field(..., description="Whether Powell hybrid solver converged successfully")
    solver_message: str = Field(..., description="Solver exit flag / diagnostics")
    solver_time_ms: float = Field(..., description="Wall-clock solver time in milliseconds")
    leverage_ratio: float = Field(..., description="Debt barrier to Asset value ratio (D / V_A)")


class ForensicInput(BaseModel):
    """
    Institutional balance sheet and financial statement data across period t and t-1.
    Enforces balance sheet sanity checks and identity relations.
    """
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

    @model_validator(mode='after')
    def validate_balance_sheet_identities(self) -> 'ForensicInput':
        if self.working_capital_t is None:
            self.working_capital_t = self.current_assets_t - self.current_liabilities_t

        # Accounting identity integrity check
        if self.current_assets_t > self.total_assets_t * 1.05:
            raise ValueError(
                f"Current assets ({self.current_assets_t}) exceeds total assets ({self.total_assets_t})"
            )
        if self.current_assets_t_minus_1 > self.total_assets_t_minus_1 * 1.05:
            raise ValueError(
                f"Prior period current assets ({self.current_assets_t_minus_1}) exceeds total assets ({self.total_assets_t_minus_1})"
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
    manipulation_probability: float = Field(..., description="Probabilistic likelihood of manipulation Phi(M)")
    is_manipulator: bool = Field(..., description="True if M-Score > -1.78 (active red flag threshold)")
    red_flags: List[str] = Field(default_factory=list, description="Specific sub-indices indicating forensic anomalies")


class AltmanResult(BaseModel):
    x1: float = Field(..., description="Working Capital / Total Assets")
    x2: float = Field(..., description="Retained Earnings / Total Assets")
    x3: float = Field(..., description="EBIT / Total Assets")
    x4: float = Field(..., description="Book Value Equity / Total Liabilities")
    z_double_prime: float = Field(..., description="Altman Z''-Score (Services & Emerging Markets)")
    zone: str = Field(..., description="Classification Zone: Safe, Grey, or Distress")
    distress_assessment: str = Field(..., description="Executive interpretation of insolvency likelihood")


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


class ForensicAnalysisResult(BaseModel):
    beneish: BeneishResult
    altman: AltmanResult
    ccc: CCCResult
    composite_forensic_risk: str = Field(..., description="Composite risk tier: Low, Medium, High, or Critical")
    summary_findings: List[str] = Field(default_factory=list, description="Forensic credit investigator bullet findings")


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


class ComprehensiveCreditRequest(BaseModel):
    company_name: str = Field(..., description="Borrower or corporate entity name")
    ticker: Optional[str] = Field(default=None, description="Equity ticker symbol if public")
    merton_input: MertonInput
    forensic_input: ForensicInput
    origination_distance_to_default: Optional[float] = Field(
        default=None,
        description="Historical DD at debt origination for IFRS 9 staging comparison. Defaults to current DD + 1.2 if None"
    )
    days_past_due: int = Field(default=0, ge=0)
    effective_interest_rate: float = Field(default=0.06, ge=0.0)
    loan_tenor_years: int = Field(default=5, ge=1, le=30)


class ComprehensiveCreditAssessment(BaseModel):
    company_name: str
    ticker: Optional[str] = None
    timestamp: str
    merton: MertonOutput
    forensics: ForensicAnalysisResult
    vasicek: VasicekOutput
    ifrs9_ecl: ECLResult
    composite_rating: str = Field(..., description="Institutional credit rating (e.g. AAA, AA, A, BBB, BB, B, CCC, D)")
    recommendation: str = Field(..., description="Institutional credit committee underwriting directive")
    executive_summary: str = Field(..., description="Comprehensive risk appraisal memo text")
