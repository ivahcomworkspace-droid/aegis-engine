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
from fastapi.middleware.gzip import GZipMiddleware
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
        BorrowerMarketData,
    )
    from aegis_engine.core.merton import solve_merton
    from aegis_engine.core.forensics import calculate_forensics
    from aegis_engine.core.vasicek import calculate_vasicek_asrf, calculate_ifrs9_ecl
    from aegis_engine.services.memo_generator import (
        generate_credit_memo_pdf,
        generate_credit_memo_html,
    )
    from aegis_engine.services.market_data import MarketDataError, company_directory, load_borrower
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
        BorrowerMarketData,
    )
    from core.merton import solve_merton
    from core.forensics import calculate_forensics
    from core.vasicek import calculate_vasicek_asrf, calculate_ifrs9_ecl
    from services.memo_generator import (
        generate_credit_memo_pdf,
        generate_credit_memo_html,
    )
    from services.market_data import MarketDataError, company_directory, load_borrower

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

# The company list for the dropdowns is about 600 KB of JSON; compressed it's under 200 KB
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def revalidate_html(request, call_next):
    """The single-page app must not be served stale from the browser cache after an update."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"IFRS 9 ECL error: {str(e)}"
        )


def _compute_composite_rating_and_recommendation(
    merton_res: MertonOutput,
    forensics_res: ForensicAnalysisResult,
    vasicek_res: VasicekOutput,
    ecl_res: ECLResult,
    use_forensics: bool = True,
) -> Tuple[str, str, str]:
    """
    Underwriting engine synthesis: maps quantitative vectors into credit ratings
    and actionable underwriting directives. With use_forensics=False (financial companies), Altman Z''
    and Beneish are left out, since both are calibrated on non-financial companies.
    """
    dd = merton_res.distance_to_default_physical
    z_double = forensics_res.altman.z_double_prime
    m_score = forensics_res.beneish.m_score
    is_manip = use_forensics and forensics_res.beneish.is_manipulator
    stage = ecl_res.assigned_stage

    def z_below(threshold: float) -> bool:
        return use_forensics and z_double < threshold

    # Rating determination
    if stage == 3 or dd < 0.5 or z_below(0.8):
        rating = "CCC/D"
        rec = "REJECT / IMPAIRMENT WATCH: High imminent default probability; trigger debt work-out."
    elif dd < 1.2 or z_below(1.10) or is_manip:
        rating = "B" if not is_manip else "B-"
        rec = "UNDERWRITE STRICT COVENANTS: High leverage, weak liquidity headroom or forensic red flags."
    elif dd < 2.0 or z_below(1.80) or stage == 2:
        rating = "BB"
        rec = "MONITOR CLOSELY: Significant increase in credit risk (SICR) observed; quarterly debt covenants required."
    elif dd < 3.2 or z_below(2.60):
        rating = "BBB"
        rec = "APPROVE / INVESTMENT GRADE: Adequate solvency buffers with standard negative pledge covenants."
    elif dd < 4.5:
        rating = "A"
        rec = "APPROVE / HIGH QUALITY: Strong balance sheet capacity, negligible refinancing headwinds."
    elif dd < 6.0:
        rating = "AA"
        rec = "PRIME QUALITY: Robust enterprise asset coverage, near-zero asset volatility risk."
    else:
        rating = "AAA"
        rec = "SUPERIOR QUALITY: Minimal solvency risk across all macroeconomic stress permutations."

    if use_forensics:
        forensic_clause = (
            f"Altman Z'' of {z_double:.2f} ({forensics_res.altman.zone}), and Beneish M-Score of {m_score:.2f} "
            f"({'MANIPULATION ALERT' if is_manip else 'CLEAN'})"
        )
    else:
        forensic_clause = "forensic scores not applied (calibrated on non-financial companies)"
    summary = (
        f"Borrower exhibits physical Distance-to-Default of {dd:.2f}σ (PD: {merton_res.default_probability_physical:.3%}), "
        f"{forensic_clause}. Classified under IFRS 9 as {ecl_res.stage_name} "
        f"with lifetime ECL allowance of ${ecl_res.total_ecl:,.0f}."
    )

    return rating, rec, summary


@app.post("/credit/comprehensive", response_model=ComprehensiveCreditAssessment, tags=["Comprehensive Engine"])
@app.post("/api/v1/credit/comprehensive", response_model=ComprehensiveCreditAssessment, tags=["Comprehensive Engine"])
async def comprehensive_assessment_endpoint(req: ComprehensiveCreditRequest) -> ComprehensiveCreditAssessment:
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
    ead = max(m_in.short_term_debt + m_in.long_term_debt, req.loan_exposure, 100_000.0)
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

    rating, rec, summary = _compute_composite_rating_and_recommendation(
        merton_res, forensics_res, vasicek_res, ecl_res, use_forensics=req.forensics_applicable
    )

    return ComprehensiveCreditAssessment(
        company_name=req.company_name,
        ticker=req.ticker,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        merton=merton_res,
        forensics=forensics_res,
        vasicek=vasicek_res,
        ifrs9_ecl=ecl_res,
        composite_rating=rating,
        recommendation=rec,
        executive_summary=summary,
        forensics_applicable=req.forensics_applicable,
    )


@app.get("/api/v1/market/borrower", response_model=BorrowerMarketData, tags=["Live Market Data"])
async def borrower_market_data_endpoint(ticker: str) -> BorrowerMarketData:
    """
    Fills the Merton inputs for a listed borrower from public data, with each value's source:
    - Equity value: latest price x SEC shares outstanding; volatility from a year of daily returns
    - Short- and long-term debt from the latest 10-Q/10-K (banks: deposits count as short-term debt)
    - Risk-free rate: US Treasury 1-year yield, continuously compounded
    - The last two annual reports as forensic inputs, scored with Beneish, Altman Z'' and CCC
    Sources fail independently; anything not filled is listed in `warnings`.
    Requires AEGIS_SEC_USER_AGENT (a name and email) because the SEC asks clients to identify themselves.
    """
    try:
        data = await load_borrower(ticker)
    except MarketDataError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    analysis = None
    if data["forensic_input"] is not None:
        analysis = await run_in_threadpool(calculate_forensics, ForensicInput(**data["forensic_input"]))
    return BorrowerMarketData(**data, forensic_analysis=analysis)


@app.get("/api/v1/market/companies", tags=["Live Market Data"])
async def company_directory_endpoint() -> Dict[str, Any]:
    """
    Every ticker in the SEC's company list, for the dashboard's ticker and company dropdowns.
    Each row is [ticker, name, exchange, automatic, cik]. `automatic` is 1 when the company filed a
    10-K or 10-Q in the last three quarters, so Load can fill its inputs; those rows come first,
    each group in the SEC's order (roughly by market value). Cached for a day.
    """
    try:
        return await company_directory()
    except MarketDataError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


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
