"""
Aegis Quantitative Credit Risk Engine API
High-throughput asynchronous FastAPI microservice exposing:
- Merton structural option solver with analytical Jacobian
- Balance sheet forensics (Beneish 8-Factor, Altman Z'', Cash Conversion Cycle)
- Vasicek ASRF macro loss engine & IFRS 9 multi-period ECL staging
- Comprehensive institutional credit assessments & PDF/HTML memorandum compilation
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from aegis_engine.schemas.credit_models import (
        MertonInput,
        MertonOutput,
        ForensicInput,
        ForensicAnalysisResult,
        VasicekInput,
        VasicekOutput,
        IFRS9StageInput,
        ECLResult,
        ComprehensiveCreditRequest,
        ComprehensiveCreditAssessment,
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
        ForensicInput,
        ForensicAnalysisResult,
        VasicekInput,
        VasicekOutput,
        IFRS9StageInput,
        ECLResult,
        ComprehensiveCreditRequest,
        ComprehensiveCreditAssessment,
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
            "Automated Institutional Credit Memorandum Generation (PDF/HTML)"
        ]
    }


@app.post("/merton/solve", response_model=MertonOutput, tags=["Structural Credit"])
@app.post("/api/v1/merton/solve", response_model=MertonOutput, tags=["Structural Credit"])
async def solve_merton_endpoint(payload: MertonInput) -> MertonOutput:
    """
    Inverts latent enterprise asset value (V_A) and asset volatility (sigma_A)
    from observable equity market cap, equity volatility, and debt default barrier.
    """
    try:
        return solve_merton(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Merton solver failure: {str(e)}"
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
        return calculate_forensics(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Forensics calculation error: {str(e)}"
        )


@app.post("/regulatory/vasicek", response_model=VasicekOutput, tags=["Regulatory Capital"])
@app.post("/api/v1/regulatory/vasicek", response_model=VasicekOutput, tags=["Regulatory Capital"])
async def vasicek_endpoint(payload: VasicekInput) -> VasicekOutput:
    """
    Calculates Vasicek Asymptotic Single Risk Factor (ASRF) conditional default probability,
    Basel corporate asset correlation rho, and regulatory capital requirements.
    """
    try:
        return calculate_vasicek_asrf(payload)
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
        return calculate_ifrs9_ecl(payload)
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
) -> tuple[str, str, str]:
    """
    Underwriting engine synthesis: maps quantitative vectors into credit ratings
    and actionable underwriting directives.
    """
    dd = merton_res.distance_to_default_physical
    z_double = forensics_res.altman.z_double_prime
    m_score = forensics_res.beneish.m_score
    is_manip = forensics_res.beneish.is_manipulator
    stage = ecl_res.assigned_stage

    # Rating determination
    if stage == 3 or dd < 0.5 or z_double < 0.8:
        rating = "CCC/D"
        rec = "REJECT / IMPAIRMENT WATCH: High imminent default probability; trigger debt work-out."
    elif dd < 1.2 or z_double < 1.10 or is_manip:
        rating = "B" if not is_manip else "B-"
        rec = "UNDERWRITE STRICT COVENANTS: High leverage, weak liquidity headroom or forensic red flags."
    elif dd < 2.0 or z_double < 1.80 or stage == 2:
        rating = "BB"
        rec = "MONITOR CLOSELY: Significant increase in credit risk (SICR) observed; quarterly debt covenants required."
    elif dd < 3.2 or z_double < 2.60:
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

    summary = (
        f"Borrower exhibits physical Distance-to-Default of {dd:.2f}σ (PD: {merton_res.default_probability_physical:.3%}), "
        f"Altman Z'' of {z_double:.2f} ({forensics_res.altman.zone}), and Beneish M-Score of {m_score:.2f} "
        f"({'MANIPULATION ALERT' if is_manip else 'CLEAN'}). Classified under IFRS 9 as {ecl_res.stage_name} "
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
    """
    # 1. Structural Merton
    merton_res = solve_merton(req.merton_input)

    # 2. Forensics
    forensics_res = calculate_forensics(req.forensic_input)

    # 3. Vasicek ASRF
    # Use physical PD from Merton as PIT PD baseline
    pit_pd = max(min(merton_res.default_probability_physical, 0.999), 0.0001)
    # Estimate total debt exposure as EAD
    ead = max(req.merton_input.short_term_debt + req.merton_input.long_term_debt, 100_000.0)
    vasicek_in = VasicekInput(
        pit_pd=pit_pd,
        lgd=req.merton_input.loss_given_default,
        ead=ead,
        maturity=req.merton_input.maturity,
    )
    vasicek_res = calculate_vasicek_asrf(vasicek_in)

    # 4. IFRS 9 ECL
    orig_dd = req.origination_distance_to_default
    if orig_dd is None:
        # If not provided, assume origination was 1.2 sigma healthier for demo purposes
        orig_dd = merton_res.distance_to_default_physical + 0.5

    # Amortization schedule
    tenor = req.loan_tenor_years
    ead_schedule = [ead * ((tenor - i) / tenor) for i in range(tenor)]

    ecl_in = IFRS9StageInput(
        current_distance_to_default=merton_res.distance_to_default_physical,
        origination_distance_to_default=orig_dd,
        current_pd=pit_pd,
        days_past_due=req.days_past_due,
        is_credit_impaired=(req.days_past_due >= 90),
        ead_schedule=ead_schedule,
        effective_interest_rate=req.effective_interest_rate,
        lifetime_periods=tenor,
    )
    ecl_res = calculate_ifrs9_ecl(ecl_in)

    rating, rec, summary = _compute_composite_rating_and_recommendation(
        merton_res, forensics_res, vasicek_res, ecl_res
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
    )


@app.post("/credit/memo/pdf", tags=["Memorandum Compilation"])
@app.post("/api/v1/credit/export-memo", tags=["Memorandum Compilation"])
@app.post("/api/v1/credit/memo/pdf", tags=["Memorandum Compilation"])
async def credit_memo_pdf_endpoint(req: ComprehensiveCreditRequest) -> Response:
    """
    Compiles an institutional-grade PDF Credit Risk Memorandum with embedded
    high-resolution vector/PNG charts and underwriting tables.
    """
    assessment = await comprehensive_assessment_endpoint(req)
    pdf_bytes = generate_credit_memo_pdf(assessment)

    filename = f"Aegis_Credit_Memo_{req.company_name.replace(' ', '_')}.pdf"
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
    """
    assessment = await comprehensive_assessment_endpoint(req)
    html_text = generate_credit_memo_html(assessment)
    return HTMLResponse(content=html_text)


# Mount static frontend directory to serve single-page application at root
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if not os.path.exists(frontend_dir):
    frontend_dir = os.path.join(os.getcwd(), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

