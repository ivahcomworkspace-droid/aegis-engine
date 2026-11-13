"""
Aegis Quantitative Credit Risk Engine API
High-throughput asynchronous FastAPI microservice exposing:
- Merton structural option solver with analytical Jacobian
- Balance sheet forensics (Beneish 8-Factor, Altman Z'', Cash Conversion Cycle)
- Vasicek ASRF macro loss engine & IFRS 9 multi-period ECL staging
- Comprehensive institutional credit assessments & PDF/HTML memorandum compilation
- Non-blocking execution via threadpool offloading
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

try:
    from aegis_engine.schemas.credit_models import (
        MertonInput,
        MertonOutput,
        MertonAnalysisRequest,
        ForensicInput,
        ForensicAnalysisRequest,
        ForensicAnalysisResult,
        VasicekInput,
        VasicekOutput,
        IFRS9StageInput,
        ECLResult,
        ComprehensiveCreditRequest,
        ComprehensiveCreditAssessment,
        ComprehensiveCreditResponse,
        AegisInternalRiskBand,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from aegis_engine.core.merton import solve_merton
    from aegis_engine.core.forensics import calculate_forensics
    from aegis_engine.core.vasicek import calculate_vasicek_asrf, calculate_ifrs9_ecl
    from aegis_engine.services.memo_generator import (
        generate_credit_memo_pdf,
        generate_credit_memo_html,
    )
except ModuleNotFoundError:
    from schemas.credit_models import (
        MertonInput,
        MertonOutput,
        MertonAnalysisRequest,
        ForensicInput,
        ForensicAnalysisRequest,
        ForensicAnalysisResult,
        VasicekInput,
        VasicekOutput,
        IFRS9StageInput,
        ECLResult,
        ComprehensiveCreditRequest,
        ComprehensiveCreditAssessment,
        ComprehensiveCreditResponse,
        AegisInternalRiskBand,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from core.merton import solve_merton
    from core.forensics import calculate_forensics
    from core.vasicek import calculate_vasicek_asrf, calculate_ifrs9_ecl
    from services.memo_generator import (
        generate_credit_memo_pdf,
        generate_credit_memo_html,
    )

app = FastAPI(
    title="Aegis Institutional Credit Risk & Forensics Engine",
    description=(
        "The defensive institutional counterpart to equity valuation platforms. "
        "Solves for asymmetric downside risk, distance-to-default, latent asset volatility inversion, "
        "forensic earnings manipulation (Beneish M-Score, Altman Z''), and regulatory capital / IFRS 9 ECL."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing engine status and computational capabilities."""
    return {
        "status": "online",
        "service": "Aegis Credit Risk Engine",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capabilities": [
            "Merton Powell Hybrid 2D Solver with Exact Analytical Jacobian",
            "Beneish 8-Factor Earnings Manipulation Detection",
            "Altman Z''-Score Emerging Markets / Services Distress Model",
            "Cash Conversion Cycle (CCC) Working Capital Dynamics",
            "Vasicek ASRF Macro Loss Distribution & Basel IRB Capital",
            "IFRS 9 / CECL Multi-Period Lifetime Expected Credit Loss (3-Stage)",
            "Automated Institutional Credit Memorandum Generation (PDF/HTML)",
            "Asynchronous Non-Blocking Threadpool Dispatch"
        ]
    }


@app.post("/merton/solve", response_model=MertonOutput, tags=["Structural Credit"])
@app.post("/api/v1/merton/solve", response_model=MertonOutput, tags=["Structural Credit"])
async def solve_merton_endpoint(payload: MertonInput) -> MertonOutput:
    """
    Inverts latent enterprise asset value (V_A) and asset volatility (sigma_A)
    from observable equity market cap, equity volatility, and debt default barrier.
    Dispatched non-blocking to worker threadpool.
    """
    try:
        return await run_in_threadpool(solve_merton, payload)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Merton solver failure: {str(e)}"
        )


@app.post("/merton/analyze", response_model=MertonOutput, tags=["Structural Credit"])
@app.post("/api/v1/merton/analyze", response_model=MertonOutput, tags=["Structural Credit"])
async def analyze_merton_endpoint(payload: MertonAnalysisRequest) -> MertonOutput:
    """
    Perimeter endpoint for structural credit analysis with strict Pydantic v2 boundaries.
    """
    m_in = MertonInput(
        equity_value=payload.equity_market_cap,
        equity_volatility=payload.equity_volatility,
        short_term_debt=payload.short_term_debt,
        long_term_debt=payload.long_term_debt,
        risk_free_rate=payload.risk_free_rate,
        drift_mu=payload.asset_drift_mu,
        maturity=payload.maturity_years,
        loss_given_default=1.0 - payload.recovery_rate
    )
    try:
        return await run_in_threadpool(solve_merton, m_in)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Merton analysis failure: {str(e)}"
        )


@app.post("/forensics/analyze", response_model=ForensicAnalysisResult, tags=["Forensics"])
@app.post("/api/v1/forensics/analyze", response_model=ForensicAnalysisResult, tags=["Forensics"])
async def analyze_forensics_endpoint(payload: ForensicInput) -> ForensicAnalysisResult:
    """
    Performs comprehensive balance sheet forensic analysis:
    - 8-Factor Beneish M-Score (threshold > -1.78)
    - Altman Z''-Score (Safe > 2.60, Grey 1.10-2.60, Distress < 1.10)
    - Cash Conversion Cycle working capital stress alert (Delta CCC > +20 days)
    """
    try:
        return await run_in_threadpool(calculate_forensics, payload)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Forensics calculation error: {str(e)}"
        )


@app.post("/forensics/audit", response_model=ForensicAnalysisResult, tags=["Forensics"])
@app.post("/api/v1/forensics/audit", response_model=ForensicAnalysisResult, tags=["Forensics"])
async def audit_forensics_endpoint(payload: ForensicAnalysisRequest) -> ForensicAnalysisResult:
    """
    Perimeter endpoint for balance sheet forensics enforcing double-entry reconciliation.
    """
    f_in = ForensicInput(
        sales_t=payload.sales_t0,
        cogs_t=payload.cogs_t0,
        receivables_t=payload.receivables_t0,
        current_assets_t=payload.current_assets_t0,
        ppe_t=payload.ppe_t0,
        total_assets_t=payload.total_assets_t0,
        depreciation_t=payload.depreciation_t0,
        sga_t=payload.sga_t0,
        net_income_t=payload.net_income_t0,
        cash_flow_operations_t=payload.operating_cash_flow_t0,
        current_liabilities_t=max(payload.total_liabilities_t0 * 0.4, 1.0),
        long_term_debt_t=payload.total_debt_t0,
        total_liabilities_t=payload.total_liabilities_t0,
        inventory_t=payload.inventory_t0,
        accounts_payable_t=payload.payables_t0,
        ebit_t=payload.ebit_t0,
        retained_earnings_t=payload.retained_earnings_t0,
        book_value_equity_t=payload.book_value_equity_t0,
        working_capital_t=payload.working_capital_t0,
        securities_t=payload.securities_t0,
        sales_t_minus_1=payload.sales_t1,
        cogs_t_minus_1=payload.cogs_t1,
        receivables_t_minus_1=payload.receivables_t1,
        current_assets_t_minus_1=payload.current_assets_t1,
        ppe_t_minus_1=payload.ppe_t1,
        total_assets_t_minus_1=payload.total_assets_t1,
        depreciation_t_minus_1=payload.depreciation_t1,
        sga_t_minus_1=payload.sga_t1,
        long_term_debt_t_minus_1=payload.total_debt_t1,
        total_liabilities_t_minus_1=max(payload.total_assets_t1 * 0.4, 1.0),
        inventory_t_minus_1=payload.inventory_t1,
        accounts_payable_t_minus_1=payload.payables_t1,
        securities_t_minus_1=payload.securities_t1,
    )
    try:
        return await run_in_threadpool(calculate_forensics, f_in)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Forensics audit failure: {str(e)}"
        )


@app.post("/regulatory/vasicek", response_model=VasicekOutput, tags=["Regulatory Capital"])
@app.post("/api/v1/regulatory/vasicek", response_model=VasicekOutput, tags=["Regulatory Capital"])
async def vasicek_endpoint(payload: VasicekInput) -> VasicekOutput:
    """
    Calculates Vasicek Asymptotic Single Risk Factor (ASRF) conditional default probability,
    Basel corporate asset correlation rho, and regulatory capital requirements.
    """
    try:
        return await run_in_threadpool(calculate_vasicek_asrf, payload)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Vasicek engine failure: {str(e)}"
        )


@app.post("/regulatory/ecl", response_model=ECLResult, tags=["IFRS 9 / CECL"])
@app.post("/api/v1/regulatory/ecl", response_model=ECLResult, tags=["IFRS 9 / CECL"])
async def ecl_endpoint(payload: IFRS9StageInput) -> ECLResult:
    """
    Evaluates IFRS 9 3-stage transitions and computes multi-period Expected Credit Loss (ECL).
    Transitions to Stage 2 upon Delta DD < -1.0 sigma (SICR) or DPD in [30, 89].
    """
    try:
        return await run_in_threadpool(calculate_ifrs9_ecl, payload)
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"IFRS 9 ECL error: {str(e)}"
        )


def _compute_internal_risk_band_and_stance(
    merton_res: MertonOutput,
    forensics_res: ForensicAnalysisResult,
    vasicek_res: VasicekOutput,
    ecl_res: ECLResult,
) -> Tuple[AegisInternalRiskBand, str, str]:
    """
    Underwriting engine synthesis: maps quantitative vectors into Aegis internal risk bands (Band 1 through Band 7)
    and non-binding indicative underwriting postures.
    """
    dd = merton_res.distance_to_default_physical
    z_double = forensics_res.altman.z_double_prime
    m_score = forensics_res.beneish.m_score
    is_manip = forensics_res.beneish.is_manipulator
    stage = ecl_res.assigned_stage

    # Internal Risk Band determination (Aegis Band 1 - 7)
    if stage == 3 or dd < 0.5 or z_double < 0.8:
        band = AegisInternalRiskBand.BAND_7
        stance = "INDICATIVE: VERY HIGH RISK / WORKOUT — Immediate loss mitigation & restructuring review required."
    elif dd < 1.2 or z_double < 1.10 or is_manip:
        band = AegisInternalRiskBand.BAND_6
        stance = "INDICATIVE: HIGH RISK / WATCHLIST — Strict covenant enforcement & collateral enhancement."
    elif dd < 2.0 or z_double < 1.80 or stage == 2:
        band = AegisInternalRiskBand.BAND_5
        stance = "INDICATIVE: ELEVATED RISK / SURVEILLANCE — Significant increase in credit risk (SICR); quarterly watch."
    elif dd < 3.2 or z_double < 2.60:
        band = AegisInternalRiskBand.BAND_4
        stance = "INDICATIVE: ACCEPTABLE RISK / MONITOR — Standard credit terms with standard negative pledge."
    elif dd < 4.5:
        band = AegisInternalRiskBand.BAND_3
        stance = "INDICATIVE: MODERATE RISK / MONITOR — Favorable repayment capacity with routine annual review."
    elif dd < 6.0:
        band = AegisInternalRiskBand.BAND_2
        stance = "INDICATIVE: LOW RISK / MONITOR — Strong solvency headroom & superior operating stability."
    else:
        band = AegisInternalRiskBand.BAND_1
        stance = "INDICATIVE: MINIMAL RISK / PRIME SOLVENCY — Exceptional asset backing & fortress liquidity coverage."

    summary = (
        f"Borrower internal credit assessment: {band.value}. "
        f"Physical Distance-to-Default: {dd:.2f}σ (PD: {merton_res.default_probability_physical:.3%}), "
        f"Altman Z'': {z_double:.2f} ({forensics_res.altman.zone}), Beneish M-Score: {m_score:.2f} "
        f"({'MANIPULATION ALERT' if is_manip else 'CLEAN'}). IFRS 9 Staging: {ecl_res.stage_name} "
        f"with ECL allowance of ${ecl_res.total_ecl:,.2f} ({ecl_res.ecl_percentage_of_ead:.2f}% EAD)."
    )

    return band, stance, summary


def _compute_composite_rating_and_recommendation(
    merton_res: MertonOutput,
    forensics_res: ForensicAnalysisResult,
    vasicek_res: VasicekOutput,
    ecl_res: ECLResult,
) -> Tuple[str, str, str]:
    """Legacy compatibility bridge mapping internal risk bands to string tuple."""
    band, stance, summary = _compute_internal_risk_band_and_stance(merton_res, forensics_res, vasicek_res, ecl_res)
    return band.value, stance, summary


@app.post("/credit/comprehensive", response_model=ComprehensiveCreditResponse, tags=["Comprehensive Engine"])
@app.post("/api/v1/credit/comprehensive", response_model=ComprehensiveCreditResponse, tags=["Comprehensive Engine"])
async def comprehensive_assessment_endpoint(req: ComprehensiveCreditRequest) -> ComprehensiveCreditResponse:
    """
    Unified end-to-end institutional credit assessment combining:
    - Merton structural asset inversion
    - Full balance sheet forensic audit (Beneish, Altman Z'', CCC)
    - Vasicek macro conditional distribution
    - IFRS 9 multi-period ECL staging schedule
    - Composite credit rating and committee directive
    - Non-blocking threadpool execution
    """
    # 1. Structural Merton Input
    if req.merton_input is not None:
        m_in = req.merton_input
    else:
        m_in = MertonInput(
            equity_value=req.equity_market_cap,
            equity_volatility=req.equity_volatility,
            short_term_debt=req.short_term_debt,
            long_term_debt=req.long_term_debt,
            risk_free_rate=req.risk_free_rate,
            drift_mu=req.asset_drift_mu,
            maturity=req.maturity_years,
            loss_given_default=1.0 - req.recovery_rate
        )

    # 2. Forensic Input
    if req.forensic_input is not None:
        f_in = req.forensic_input
    else:
        sales = req.revenue
        cogs = req.cogs
        ta = req.total_assets
        tl = req.total_liabilities
        bve = req.book_value_equity
        wc = req.working_capital
        re = req.retained_earnings
        ebit = req.ebit
        rec = req.receivables
        inv = req.inventory
        pay = req.payables

        if req.simulate_beneish_manipulation:
            sales_prev = sales * 0.70
            rec_prev = rec * 0.30
            cogs_prev = cogs * 0.50
            cfo_curr = ebit * 0.10
        else:
            sales_prev = sales * 0.95
            rec_prev = rec * 0.95
            cogs_prev = cogs * 0.95
            cfo_curr = ebit * 0.85

        if req.simulate_working_capital_stress:
            rec_prev_ccc = rec * 0.50
            inv_prev = inv * 0.50
        else:
            rec_prev_ccc = rec_prev
            inv_prev = inv * 0.95

        f_in = ForensicInput(
            sales_t=sales,
            cogs_t=cogs,
            receivables_t=rec,
            current_assets_t=max(wc + tl * 0.5, rec + inv + 10.0),
            ppe_t=max(ta * 0.4, 10.0),
            total_assets_t=ta,
            depreciation_t=max(ebit * 0.2, 5.0),
            sga_t=max(sales * 0.15, 5.0),
            net_income_t=ebit * 0.7,
            cash_flow_operations_t=cfo_curr,
            current_liabilities_t=max(tl * 0.4, 5.0),
            long_term_debt_t=max(tl * 0.6, 0.0),
            total_liabilities_t=tl,
            inventory_t=inv,
            accounts_payable_t=pay,
            ebit_t=ebit,
            retained_earnings_t=re,
            book_value_equity_t=bve,
            working_capital_t=wc,
            sales_t_minus_1=sales_prev,
            cogs_t_minus_1=cogs_prev,
            receivables_t_minus_1=rec_prev_ccc,
            current_assets_t_minus_1=max(ta * 0.95 * 0.4, 10.0),
            ppe_t_minus_1=max(ta * 0.95 * 0.4, 10.0),
            total_assets_t_minus_1=ta * 0.95,
            depreciation_t_minus_1=max(ebit * 0.2 * 0.95, 5.0),
            sga_t_minus_1=max(sales_prev * 0.15, 5.0),
            long_term_debt_t_minus_1=max(tl * 0.6 * 0.95, 0.0),
            total_liabilities_t_minus_1=tl * 0.95,
            inventory_t_minus_1=inv_prev,
            accounts_payable_t_minus_1=pay * 0.95
        )

    # 3. Offload core analytics to threadpool
    merton_res = await run_in_threadpool(solve_merton, m_in)
    forensics_res = await run_in_threadpool(calculate_forensics, f_in)

    # 4. Vasicek ASRF
    pit_pd = max(min(merton_res.default_probability_physical, 0.999), 0.0001)
    # Fix A1: Use explicit loan exposure provided in the request without artificial floor
    ead = float(req.loan_exposure)
    vasicek_in = VasicekInput(
        pit_pd=pit_pd,
        lgd=m_in.loss_given_default,
        ead=ead,
        maturity=m_in.maturity,
    )
    vasicek_res = await run_in_threadpool(calculate_vasicek_asrf, vasicek_in)

    # 5. IFRS 9 ECL
    orig_dd = req.origination_distance_to_default
    if orig_dd is None:
        orig_dd = merton_res.distance_to_default_physical + 0.5

    tenor = max(req.loan_tenor_years, int(req.maturity_years), 1)
    ead_schedule = [ead * ((tenor - i) / tenor) for i in range(tenor)]

    ecl_in = IFRS9StageInput(
        current_distance_to_default=merton_res.distance_to_default_physical,
        origination_distance_to_default=orig_dd,
        current_pd=pit_pd,
        days_past_due=req.days_past_due,
        is_credit_impaired=(req.days_past_due >= 90),
        ead_schedule=ead_schedule,
        effective_interest_rate=req.effective_interest_rate,
    )
    ecl_res = await run_in_threadpool(calculate_ifrs9_ecl, ecl_in)

    band, stance, summary = _compute_internal_risk_band_and_stance(
        merton_res, forensics_res, vasicek_res, ecl_res
    )

    return ComprehensiveCreditResponse(
        ticker=req.ticker or "",
        company_name=req.company_name,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        internal_risk_band=band.value,
        indicative_stance=stance,
        credit_rating=band.value,
        recommendation=stance,
        regulatory_disclaimer=DEFAULT_REGULATORY_DISCLAIMER,
        merton=merton_res,
        forensics=forensics_res,
        regulatory_capital=vasicek_res,
        ifrs9_ecl=ecl_res,
        composite_rating=band.value,
        vasicek=vasicek_res,
        executive_summary=summary,
    )


@app.post("/credit/memo/pdf", tags=["Memorandum Compilation"])
@app.post("/api/v1/credit/export-memo", tags=["Memorandum Compilation"])
@app.post("/api/v1/credit/memo/pdf", tags=["Memorandum Compilation"])
async def credit_memo_pdf_endpoint(req: ComprehensiveCreditRequest) -> Response:
    """
    Compiles an institutional-grade PDF Credit Risk Memorandum with embedded
    high-resolution vector/PNG charts and underwriting tables.
    Runs non-blocking in threadpool.
    """
    assessment = await comprehensive_assessment_endpoint(req)
    pdf_bytes = await run_in_threadpool(generate_credit_memo_pdf, assessment)

    safe_name = req.company_name.replace(' ', '_').replace('/', '_')
    filename = f"Aegis_Credit_Memo_{safe_name}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/credit/memo/html", response_class=HTMLResponse, tags=["Memorandum Compilation"])
@app.post("/api/v1/credit/memo/html", response_class=HTMLResponse, tags=["Memorandum Compilation"])
async def credit_memo_html_endpoint(req: ComprehensiveCreditRequest) -> HTMLResponse:
    """
    Generates interactive HTML version of the institutional credit memorandum.
    Runs non-blocking in threadpool.
    """
    assessment = await comprehensive_assessment_endpoint(req)
    html_text = await run_in_threadpool(generate_credit_memo_html, assessment)
    return HTMLResponse(content=html_text)


# Mount static frontend directory to serve single-page application at root
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if not os.path.exists(frontend_dir):
    frontend_dir = os.path.join(os.getcwd(), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
