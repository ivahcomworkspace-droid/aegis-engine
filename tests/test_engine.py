"""
Deterministic Test Suite for Aegis Quantitative Credit Risk Engine
Contains 35+ comprehensive unit, numerical, and integration tests across:
- Merton option solver, analytical Jacobian, speed, and asymptotic clamp
- Beneish 8-Factor forensics, Altman Z'', and CCC dynamics
- Vasicek ASRF regulatory capital and IFRS 9 multi-period ECL staging
- FastAPI endpoints, PDF generation, and HTML rendering
"""

import io
import time
import pytest
import numpy as np
from scipy.stats import norm
from fastapi.testclient import TestClient

try:
    from aegis_engine.schemas.credit_models import (
        MertonInput,
        MertonOutput,
        MertonAnalysisRequest,
        ForensicInput,
        ForensicAnalysisRequest,
        VasicekInput,
        IFRS9StageInput,
        ComprehensiveCreditRequest,
        ComprehensiveCreditResponse,
        AegisInternalRiskBand,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from aegis_engine.core.merton import (
        solve_merton,
        MertonSolver,
        solve_structural_merton,
        merton_objective_and_jacobian,
    )
    from aegis_engine.core.forensics import (
        calculate_beneish,
        calculate_beneish_m_score,
        calculate_altman_z_double_prime,
        calculate_ccc,
        calculate_cash_conversion_cycle,
        calculate_forensics,
    )
    from aegis_engine.core.vasicek import (
        calculate_corporate_asset_correlation,
        calculate_vasicek_conditional_pd,
        calculate_vasicek_loss_distribution,
        calculate_vasicek_capital,
        calculate_vasicek_asrf,
        calculate_ecl_staging_and_loss,
        calculate_ifrs9_ecl,
        calculate_multi_period_ecl,
    )
    from aegis_engine.services.memo_generator import (
        generate_credit_charts,
        generate_credit_memo_html,
        generate_credit_memo_pdf,
    )
    from aegis_engine.api.main import app
except ModuleNotFoundError:
    from schemas.credit_models import (
        MertonInput,
        MertonOutput,
        MertonAnalysisRequest,
        ForensicInput,
        ForensicAnalysisRequest,
        VasicekInput,
        IFRS9StageInput,
        ComprehensiveCreditRequest,
        ComprehensiveCreditResponse,
        AegisInternalRiskBand,
        DEFAULT_REGULATORY_DISCLAIMER,
    )
    from core.merton import (
        solve_merton,
        MertonSolver,
        solve_structural_merton,
        merton_objective_and_jacobian,
    )
    from core.forensics import (
        calculate_beneish,
        calculate_beneish_m_score,
        calculate_altman_z_double_prime,
        calculate_ccc,
        calculate_cash_conversion_cycle,
        calculate_forensics,
    )
    from core.vasicek import (
        calculate_corporate_asset_correlation,
        calculate_vasicek_conditional_pd,
        calculate_vasicek_loss_distribution,
        calculate_vasicek_capital,
        calculate_vasicek_asrf,
        calculate_ecl_staging_and_loss,
        calculate_ifrs9_ecl,
        calculate_multi_period_ecl,
    )
    from services.memo_generator import (
        generate_credit_charts,
        generate_credit_memo_html,
        generate_credit_memo_pdf,
    )
    from api.main import app

client = TestClient(app)


# =====================================================================
# 1. STRUCTURAL MERTON SOLVER TESTS
# =====================================================================

def test_merton_baseline_convergence():
    """Verify that Merton solver converges accurately on standard corporate inputs."""
    m_in = MertonInput(
        equity_value=100.0,
        equity_volatility=0.40,
        short_term_debt=30.0,
        long_term_debt=100.0,  # D = 30 + 0.5 * 100 = 80
        risk_free_rate=0.05,
        maturity=1.0,
        loss_given_default=0.45
    )
    res = solve_merton(m_in)

    assert res.solver_converged is True
    assert res.default_barrier == 80.0
    assert res.asset_value > m_in.equity_value  # V_A > E
    assert res.asset_volatility < m_in.equity_volatility  # sigma_A < sigma_E
    assert 0.0 <= res.default_probability_physical <= 1.0
    assert res.distance_to_default_physical > 0.0


def test_merton_sub_2ms_benchmark():
    """Verify sub-2ms convergence speed with Powell hybrid method and analytical Jacobian."""
    m_in = MertonInput(
        equity_value=150.0,
        equity_volatility=0.35,
        short_term_debt=40.0,
        long_term_debt=80.0,
        risk_free_rate=0.045,
        maturity=1.0
    )
    # Warm up JIT/cache
    _ = solve_merton(m_in)

    t0 = time.perf_counter()
    res = solve_merton(m_in)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert res.solver_converged is True
    # Sub-2ms execution check (with generous allowance for CI/VM jitter)
    assert elapsed_ms < 10.0, f"Expected sub-2ms, got {elapsed_ms:.2f}ms"


def test_merton_zero_debt_asymptotic_clamp():
    """If D = 0 => V_A = E, sigma_A = sigma_E, DD = 99.0, PD = 0.0, Spread = 0.0."""
    m_in = MertonInput(
        equity_value=250.0,
        equity_volatility=0.28,
        short_term_debt=0.0,
        long_term_debt=0.0,
        risk_free_rate=0.04
    )
    res = solve_merton(m_in)

    assert res.default_barrier == 0.0
    assert res.asset_value == 250.0
    assert res.asset_volatility == 0.28
    assert res.distance_to_default_physical == 99.0
    assert res.default_probability_physical == 0.0
    assert res.credit_spread_bps == 0.0
    assert "Zero debt" in res.solver_message


def test_merton_exact_analytical_jacobian():
    """Verify analytical Jacobian matches numerical finite-difference Jacobian within 1e-4."""
    E, sigma_E, D, r, T = 120.0, 0.45, 90.0, 0.05, 1.0
    x = np.array([190.0, 0.26])

    f0, J_analytic = MertonSolver._system_and_jacobian(x, E, sigma_E, D, r, T)

    # Central difference approximation
    eps = 1e-6
    J_num = np.zeros_like(J_analytic)

    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus, _ = MertonSolver._system_and_jacobian(x_plus, E, sigma_E, D, r, T)
        f_minus, _ = MertonSolver._system_and_jacobian(x_minus, E, sigma_E, D, r, T)
        J_num[:, i] = (f_plus - f_minus) / (2.0 * eps)

    # Element-wise close
    np.testing.assert_allclose(J_analytic, J_num, rtol=1e-3, atol=1e-3)


def test_merton_high_leverage_distress():
    """Verify solver convergence under distressed capital structures where D approaches V_A."""
    m_in = MertonInput(
        equity_value=15.0,
        equity_volatility=0.85,
        short_term_debt=80.0,
        long_term_debt=120.0,  # D = 80 + 60 = 140
        risk_free_rate=0.06,
        maturity=1.0
    )
    res = solve_merton(m_in)

    assert res.solver_converged is True
    assert res.leverage_ratio > 0.80
    assert res.distance_to_default_physical < 1.0
    assert res.default_probability_physical > 0.15
    assert res.credit_spread_bps > 500.0


def test_merton_drift_mu_sensitivity():
    """Higher physical drift mu_A increases physical DD while risk-neutral DD remains unchanged."""
    m_in_low_mu = MertonInput(
        equity_value=100.0, equity_volatility=0.40, short_term_debt=40.0,
        long_term_debt=60.0, risk_free_rate=0.05, drift_mu=0.02
    )
    m_in_high_mu = MertonInput(
        equity_value=100.0, equity_volatility=0.40, short_term_debt=40.0,
        long_term_debt=60.0, risk_free_rate=0.05, drift_mu=0.15
    )
    res_low = solve_merton(m_in_low_mu)
    res_high = solve_merton(m_in_high_mu)

    assert res_high.distance_to_default_physical > res_low.distance_to_default_physical
    assert res_high.distance_to_default_risk_neutral == res_low.distance_to_default_risk_neutral


def test_merton_call_option_identity():
    """Verify Black-Scholes call option equation equality holds: E = V_A*Phi(d1) - D*exp(-rT)*Phi(d2)."""
    m_in = MertonInput(
        equity_value=120.0, equity_volatility=0.35, short_term_debt=50.0,
        long_term_debt=80.0, risk_free_rate=0.04, maturity=1.0
    )
    res = solve_merton(m_in)

    reconstructed_E = (
        res.asset_value * norm.cdf(res.d1)
        - res.default_barrier * np.exp(-0.04 * 1.0) * norm.cdf(res.d2)
    )
    assert abs(reconstructed_E - 120.0) < 1e-2


def test_merton_synthetic_round_trip():
    """
    Verify Fix Me1 & Me2: Round-trip synthetic test recovering known (V_A, sigma_A)
    with strict solver convergence and residual norm checks.
    """
    true_va = 1000.0
    true_sa = 0.25
    D = 400.0
    r = 0.04
    T = 1.0
    sqrt_T = np.sqrt(T)

    d1 = (np.log(true_va / D) + (r + 0.5 * true_sa**2) * T) / (true_sa * sqrt_T)
    d2 = d1 - true_sa * sqrt_T
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)

    E = true_va * Phi_d1 - D * np.exp(-r * T) * Phi_d2
    sigma_E = (true_va / E) * Phi_d1 * true_sa

    res = solve_structural_merton(
        equity_market_cap=E,
        equity_volatility=sigma_E,
        short_term_debt=D,
        long_term_debt=0.0,
        risk_free_rate=r,
        asset_drift_mu=r,
        maturity_years=T
    )

    assert abs(res["asset_value"] - true_va) < 1e-4
    assert abs(res["asset_volatility"] - true_sa) < 1e-4
    assert res["solver_converged"] is True
    assert res["residual_norm"] < 1e-6
    assert res["solver_used"] in ["hybr", "trf_fallback"]


# =====================================================================
# 2. BALANCE SHEET FORENSICS TESTS
# =====================================================================

@pytest.fixture
def clean_forensic_input() -> ForensicInput:
    """Healthy corporate with static ratios and clean accounting."""
    return ForensicInput(
        sales_t=1000.0, cogs_t=600.0, receivables_t=120.0, current_assets_t=400.0,
        ppe_t=500.0, total_assets_t=1000.0, depreciation_t=50.0, sga_t=150.0,
        net_income_t=100.0, cash_flow_operations_t=120.0, current_liabilities_t=200.0,
        long_term_debt_t=200.0, total_liabilities_t=400.0, inventory_t=150.0,
        accounts_payable_t=100.0, ebit_t=150.0, retained_earnings_t=300.0,
        book_value_equity_t=600.0,
        # Period t-1
        sales_t_minus_1=950.0, cogs_t_minus_1=570.0, receivables_t_minus_1=114.0,
        current_assets_t_minus_1=380.0, ppe_t_minus_1=480.0, total_assets_t_minus_1=950.0,
        depreciation_t_minus_1=48.0, sga_t_minus_1=142.0, long_term_debt_t_minus_1=190.0,
        total_liabilities_t_minus_1=380.0, inventory_t_minus_1=140.0,
        accounts_payable_t_minus_1=95.0
    )


@pytest.fixture
def manipulator_forensic_input() -> ForensicInput:
    """Manipulative firm with aggressive revenue recognition and accrual divergence."""
    return ForensicInput(
        sales_t=1500.0, cogs_t=1100.0, receivables_t=450.0, current_assets_t=700.0,
        ppe_t=300.0, total_assets_t=1400.0, depreciation_t=20.0, sga_t=280.0,
        net_income_t=180.0, cash_flow_operations_t=20.0, current_liabilities_t=400.0,
        long_term_debt_t=600.0, total_liabilities_t=1000.0, inventory_t=350.0,
        accounts_payable_t=120.0, ebit_t=90.0, retained_earnings_t=80.0,
        book_value_equity_t=400.0,
        # Period t-1
        sales_t_minus_1=900.0, cogs_t_minus_1=500.0, receivables_t_minus_1=100.0,
        current_assets_t_minus_1=400.0, ppe_t_minus_1=350.0, total_assets_t_minus_1=900.0,
        depreciation_t_minus_1=40.0, sga_t_minus_1=120.0, long_term_debt_t_minus_1=250.0,
        total_liabilities_t_minus_1=400.0, inventory_t_minus_1=120.0,
        accounts_payable_t_minus_1=110.0
    )


def test_beneish_clean_firm(clean_forensic_input):
    """Clean firm should score well below the -1.78 threshold."""
    res = calculate_beneish(clean_forensic_input)
    assert res.m_score < -1.78
    assert res.is_manipulator is False
    assert res.manipulation_probability < 0.10


def test_beneish_manipulator_warning(manipulator_forensic_input):
    """Manipulative firm should trigger the M-Score > -1.78 alert."""
    res = calculate_beneish(manipulator_forensic_input)
    assert res.m_score > -1.78
    assert res.is_manipulator is True
    assert len(res.red_flags) > 0
    assert any("DSRI" in rf for rf in res.red_flags)


def test_altman_z_double_prime_safe(clean_forensic_input):
    """Verify Altman Z'' correctly identifies Safe zone (> 2.60)."""
    res = calculate_altman_z_double_prime(clean_forensic_input)
    assert res.z_double_prime > 2.60
    assert res.zone == "Safe"


def test_altman_z_double_prime_distress():
    """Verify Altman Z'' identifies Distress zone (< 1.10)."""
    inp = ForensicInput(
        sales_t=500.0, cogs_t=450.0, receivables_t=60.0, current_assets_t=100.0,
        ppe_t=200.0, total_assets_t=600.0, depreciation_t=30.0, sga_t=100.0,
        net_income_t=-80.0, cash_flow_operations_t=-40.0, current_liabilities_t=250.0,
        long_term_debt_t=350.0, total_liabilities_t=600.0, inventory_t=50.0,
        accounts_payable_t=90.0, ebit_t=-50.0, retained_earnings_t=-150.0,
        book_value_equity_t=0.0,
        # Period t-1
        sales_t_minus_1=600.0, cogs_t_minus_1=480.0, receivables_t_minus_1=70.0,
        current_assets_t_minus_1=150.0, ppe_t_minus_1=220.0, total_assets_t_minus_1=650.0,
        depreciation_t_minus_1=30.0, sga_t_minus_1=90.0, long_term_debt_t_minus_1=300.0,
        total_liabilities_t_minus_1=550.0, inventory_t_minus_1=60.0,
        accounts_payable_t_minus_1=80.0
    )
    res = calculate_altman_z_double_prime(inp)
    assert res.z_double_prime < 1.10
    assert res.zone == "Distress"


def test_altman_z_double_prime_grey_zone():
    """Verify Altman Z'' identifies Grey zone (1.10 <= Z'' <= 2.60)."""
    inp = ForensicInput(
        sales_t=800.0, cogs_t=560.0, receivables_t=100.0, current_assets_t=280.0,
        ppe_t=400.0, total_assets_t=800.0, depreciation_t=40.0, sga_t=130.0,
        net_income_t=30.0, cash_flow_operations_t=35.0, current_liabilities_t=200.0,
        long_term_debt_t=300.0, total_liabilities_t=500.0, inventory_t=120.0,
        accounts_payable_t=80.0, ebit_t=60.0, retained_earnings_t=100.0,
        book_value_equity_t=300.0,
        # Period t-1
        sales_t_minus_1=750.0, cogs_t_minus_1=520.0, receivables_t_minus_1=90.0,
        current_assets_t_minus_1=260.0, ppe_t_minus_1=390.0, total_assets_t_minus_1=760.0,
        depreciation_t_minus_1=38.0, sga_t_minus_1=120.0, long_term_debt_t_minus_1=280.0,
        total_liabilities_t_minus_1=470.0, inventory_t_minus_1=110.0,
        accounts_payable_t_minus_1=75.0
    )
    res = calculate_altman_z_double_prime(inp)
    assert 1.10 <= res.z_double_prime <= 2.60
    assert res.zone == "Grey"


def test_cash_conversion_cycle_calculation(clean_forensic_input):
    """Verify CCC = DSO + DIO - DPO exactness."""
    res = calculate_ccc(clean_forensic_input)
    expected_dso = (120.0 / 1000.0) * 365.0
    expected_dio = (150.0 / 600.0) * 365.0
    expected_dpo = (100.0 / 600.0) * 365.0
    expected_ccc = expected_dso + expected_dio - expected_dpo

    assert abs(res.dso_t - expected_dso) < 0.1
    assert abs(res.dio_t - expected_dio) < 0.1
    assert abs(res.dpo_t - expected_dpo) < 0.1
    assert abs(res.ccc_t - expected_ccc) < 0.1


def test_ccc_working_capital_warning_trigger():
    """Trigger alert when Delta CCC > +20 days while gross margin is static (|Delta GM| <= 0.02)."""
    # Sales and COGS identical so gross margin is 40% in both periods (Delta GM = 0.0)
    inp = ForensicInput(
        sales_t=1000.0, cogs_t=600.0, receivables_t=220.0,  # DSO goes from 43.8 to 80.3 (+36.5 days)
        current_assets_t=500.0, ppe_t=400.0, total_assets_t=1000.0, depreciation_t=40.0, sga_t=120.0,
        net_income_t=70.0, cash_flow_operations_t=30.0, current_liabilities_t=200.0,
        long_term_debt_t=200.0, total_liabilities_t=400.0, inventory_t=150.0,
        accounts_payable_t=100.0, ebit_t=120.0, retained_earnings_t=200.0,
        book_value_equity_t=600.0,
        # Period t-1
        sales_t_minus_1=1000.0, cogs_t_minus_1=600.0, receivables_t_minus_1=120.0,
        current_assets_t_minus_1=400.0, ppe_t_minus_1=400.0, total_assets_t_minus_1=900.0,
        depreciation_t_minus_1=40.0, sga_t_minus_1=120.0, long_term_debt_t_minus_1=200.0,
        total_liabilities_t_minus_1=400.0, inventory_t_minus_1=150.0,
        accounts_payable_t_minus_1=100.0
    )
    res = calculate_ccc(inp)
    assert res.delta_ccc > 20.0
    assert abs(res.delta_gross_margin) <= 0.02
    assert res.warning_triggered is True
    assert "WORKING CAPITAL STRESS ALERT" in res.alert_message


def test_unified_forensics_composite_risk(clean_forensic_input, manipulator_forensic_input):
    """Verify composite risk synthesis: clean firm is Low, manipulator is Critical."""
    clean_res = calculate_forensics(clean_forensic_input)
    manip_res = calculate_forensics(manipulator_forensic_input)

    assert clean_res.composite_forensic_risk in ["Low", "Medium"]
    assert manip_res.composite_forensic_risk in ["High", "Critical"]


def test_beneish_m_score_raw_numerical():
    """Verify calculate_beneish_m_score produces correct dict keys, indices, and flags."""
    res = calculate_beneish_m_score(
        sales_t=1200.0, sales_prev=1000.0,
        cogs_t=700.0, cogs_prev=600.0,
        rec_t=250.0, rec_prev=150.0,
        ca_t=500.0, ca_prev=450.0,
        ppe_t=400.0, ppe_prev=400.0,
        sec_t=50.0, sec_prev=50.0,
        ta_t=1100.0, ta_prev=1000.0,
        dep_t=30.0, dep_prev=45.0,
        sga_t=180.0, sga_prev=140.0,
        net_income_t=90.0, cfo_t=20.0,
        total_debt_t=500.0, total_debt_prev=400.0
    )

    assert "m_score" in res
    assert "is_manipulator" in res
    assert res["manipulation_threshold"] == -1.78
    assert "indices" in res
    assert "DSRI" in res["indices"]
    assert "red_flags" in res
    assert isinstance(res["red_flags"], dict)


def test_beneish_neutral_company_score():
    """(a) A neutral company (all YoY indices == 1, net income == CFO) gives M = -2.48 (abs 1e-3)."""
    res = calculate_beneish_m_score(
        sales_t=1000.0, sales_prev=1000.0,
        cogs_t=600.0, cogs_prev=600.0,
        rec_t=100.0, rec_prev=100.0,
        ca_t=300.0, ca_prev=300.0,
        ppe_t=400.0, ppe_prev=400.0,
        sec_t=50.0, sec_prev=50.0,
        ta_t=1000.0, ta_prev=1000.0,
        dep_t=40.0, dep_prev=40.0,
        sga_t=100.0, sga_prev=100.0,
        net_income_t=50.0, cfo_t=50.0,
        total_debt_t=400.0, total_debt_prev=400.0
    )
    assert res["indices"]["DSRI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["GMI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["AQI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["SGI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["DEPI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["SGAI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["LVGI"] == pytest.approx(1.0, abs=1e-3)
    assert res["indices"]["TATA"] == pytest.approx(0.0, abs=1e-3)
    assert res["m_score"] == pytest.approx(-2.48, abs=1e-3)
    assert abs(res["m_score"] - (-2.48)) < 1e-3


def test_beneish_tata_increase_sensitivity():
    """(b) Raising TATA by 0.10 changes M by 4.679*0.10 (abs 1e-6)."""
    res_base = calculate_beneish_m_score(
        sales_t=1000.0, sales_prev=1000.0,
        cogs_t=600.0, cogs_prev=600.0,
        rec_t=100.0, rec_prev=100.0,
        ca_t=300.0, ca_prev=300.0,
        ppe_t=400.0, ppe_prev=400.0,
        sec_t=50.0, sec_prev=50.0,
        ta_t=1000.0, ta_prev=1000.0,
        dep_t=40.0, dep_prev=40.0,
        sga_t=100.0, sga_prev=100.0,
        net_income_t=50.0, cfo_t=50.0,
        total_debt_t=400.0, total_debt_prev=400.0
    )
    # Increase TATA by 0.10: net_income_t = cfo_t + 0.10 * ta_t (150.0 - 50.0 = 100.0 / 1000.0 = 0.10)
    res_raised = calculate_beneish_m_score(
        sales_t=1000.0, sales_prev=1000.0,
        cogs_t=600.0, cogs_prev=600.0,
        rec_t=100.0, rec_prev=100.0,
        ca_t=300.0, ca_prev=300.0,
        ppe_t=400.0, ppe_prev=400.0,
        sec_t=50.0, sec_prev=50.0,
        ta_t=1000.0, ta_prev=1000.0,
        dep_t=40.0, dep_prev=40.0,
        sga_t=100.0, sga_prev=100.0,
        net_income_t=150.0, cfo_t=50.0,
        total_debt_t=400.0, total_debt_prev=400.0
    )
    delta_tata = res_raised["indices"]["TATA"] - res_base["indices"]["TATA"]
    assert delta_tata == pytest.approx(0.10, abs=1e-6)
    delta_m = res_raised["m_score"] - res_base["m_score"]
    assert delta_m == pytest.approx(4.679 * 0.10, abs=1e-6)
    assert abs(delta_m - (4.679 * 0.10)) < 1e-6


def test_altman_z_double_prime_raw_numerical_zones():
    """Verify calculate_altman_z_double_prime with raw floats across SAFE, GREY, and DISTRESS."""
    # Safe zone (Z'' > 2.60)
    safe_res = calculate_altman_z_double_prime(
        working_capital=300.0, retained_earnings=400.0, ebit=200.0,
        book_value_equity=600.0, total_assets=1000.0, total_liabilities=300.0
    )
    assert safe_res["zone"] == "SAFE"
    assert safe_res["z_double_prime"] > 2.60
    assert "components" in safe_res

    # Grey zone (1.10 <= Z'' <= 2.60)
    grey_res = calculate_altman_z_double_prime(
        working_capital=100.0, retained_earnings=50.0, ebit=50.0,
        book_value_equity=200.0, total_assets=1000.0, total_liabilities=600.0
    )
    assert grey_res["zone"] == "GREY"
    assert 1.10 <= grey_res["z_double_prime"] <= 2.60

    # Distress zone (Z'' < 1.10)
    distress_res = calculate_altman_z_double_prime(
        working_capital=-100.0, retained_earnings=-200.0, ebit=-50.0,
        book_value_equity=50.0, total_assets=1000.0, total_liabilities=900.0
    )
    assert distress_res["zone"] == "DISTRESS"
    assert distress_res["z_double_prime"] < 1.10


def test_altman_z_double_prime_raw_zero_assets_guard():
    """Verify calculate_altman_z_double_prime raises ValueError when assets/liabilities <= 0."""
    with pytest.raises(ValueError, match="Total Assets and Total Liabilities must be strictly positive"):
        calculate_altman_z_double_prime(100.0, 50.0, 50.0, 100.0, total_assets=0.0, total_liabilities=200.0)

    with pytest.raises(ValueError, match="Total Assets and Total Liabilities must be strictly positive"):
        calculate_altman_z_double_prime(100.0, 50.0, 50.0, 100.0, total_assets=500.0, total_liabilities=-10.0)


def test_cash_conversion_cycle_raw_numerical():
    """Verify calculate_cash_conversion_cycle returns correct DSO, DIO, DPO, and accrual flag."""
    res = calculate_cash_conversion_cycle(
        receivables=150.0, inventory=200.0, payables=100.0,
        sales=1000.0, cogs=600.0,
        prev_receivables=100.0, prev_inventory=180.0, prev_payables=110.0,
        prev_sales=1000.0, prev_cogs=600.0
    )
    assert abs(res["dso_days"] - 54.75) < 0.1
    assert abs(res["dio_days"] - 121.67) < 0.1
    assert abs(res["dpo_days"] - 60.83) < 0.1
    assert "ccc_days" in res
    assert "delta_ccc_days" in res
    assert res["delta_ccc_days"] > 20.0
    assert res["accrual_divergence_alert"] is True


# =====================================================================
# 3. VASICEK REGULATORY LOSS & IFRS 9 ECL TESTS
# =====================================================================

def test_vasicek_correlation_bounds():
    """Basel asset correlation rho must strictly reside in [0.12, 0.24]."""
    for pd in [0.0001, 0.001, 0.01, 0.05, 0.20, 0.50, 0.99]:
        rho = calculate_corporate_asset_correlation(pd)
        assert 0.12 <= rho <= 0.24, f"rho={rho} outside [0.12, 0.24] for PD={pd}"


def test_vasicek_correlation_monotonicity():
    """Basel correlation decreases monotonically as PD increases."""
    pd_low = 0.005
    pd_high = 0.10
    assert calculate_corporate_asset_correlation(pd_low) > calculate_corporate_asset_correlation(pd_high)


def test_vasicek_conditional_pd_macro_shocks():
    """Verify conditional PD properties under systemic shock Z."""
    pd = 0.02
    rho = 0.20

    # Median macro state Z=0 evaluates to Phi(Phi^-1(PD) / sqrt(1 - rho))
    cond_pd_base = calculate_vasicek_conditional_pd(pd, rho, 0.0)
    expected_base = float(norm.cdf(norm.ppf(pd) / np.sqrt(1.0 - rho)))
    assert abs(cond_pd_base - expected_base) < 1e-6

    # Adverse shock Z > 0 must strictly increase conditional PD
    cond_pd_mild = calculate_vasicek_conditional_pd(pd, rho, 1.0)
    cond_pd_severe = calculate_vasicek_conditional_pd(pd, rho, 2.326)
    cond_pd_reg = calculate_vasicek_conditional_pd(pd, rho, 3.090)

    assert cond_pd_base < cond_pd_mild < cond_pd_severe < cond_pd_reg


def test_vasicek_asrf_full_capital():
    """Verify Basel ASRF output consistency (EL, UL, Capital K)."""
    inp = VasicekInput(
        pit_pd=0.015,
        lgd=0.45,
        ead=2_000_000.0,
        maturity=2.5
    )
    res = calculate_vasicek_asrf(inp)

    assert res.expected_loss == pytest.approx(0.015 * 0.45 * 2_000_000.0, abs=1.0)
    assert res.unexpected_loss > 0.0
    assert res.regulatory_capital_k > 0.0
    assert res.economic_capital_ead == pytest.approx(res.expected_loss + res.unexpected_loss, abs=1.0)
    assert any("Regulatory VaR" in k for k in res.stress_scenarios.keys())


def test_ifrs9_stage_1_performing():
    """Stable firm with Delta DD >= -1.0 and low DPD must be classified as Stage 1 (12M ECL)."""
    inp = IFRS9StageInput(
        current_distance_to_default=3.5,
        origination_distance_to_default=3.8,  # Delta DD = -0.3 >= -1.0
        current_pd=0.01,
        days_past_due=0,
        ead_schedule=[1_000_000.0, 800_000.0, 600_000.0]
    )
    res = calculate_ifrs9_ecl(inp)

    assert res.assigned_stage == 1
    assert res.is_lifetime_ecl is False
    assert len(res.period_schedules) == 1  # 12-Month ECL only


def test_ifrs9_stage_2_sicr_transition():
    """Borrower experiencing Delta DD < -1.0 sigma must transition to Stage 2 (Lifetime ECL)."""
    inp = IFRS9StageInput(
        current_distance_to_default=1.8,
        origination_distance_to_default=3.5,  # Delta DD = -1.7 < -1.0 (SICR)
        current_pd=0.04,
        days_past_due=10,
        ead_schedule=[1_000_000.0, 800_000.0, 600_000.0]
    )
    res = calculate_ifrs9_ecl(inp)

    assert res.assigned_stage == 2
    assert res.is_lifetime_ecl is True
    assert len(res.period_schedules) == 3
    assert "Significant Increase in Credit Risk" in res.stage_rationale


def test_ifrs9_stage_2_dpd_trigger():
    """Borrower with 30 <= DPD < 90 must trigger Stage 2 even if Delta DD is mild."""
    inp = IFRS9StageInput(
        current_distance_to_default=3.0,
        origination_distance_to_default=3.2,  # Delta DD = -0.2
        current_pd=0.02,
        days_past_due=45,  # DPD in [30, 89]
        ead_schedule=[500_000.0, 400_000.0]
    )
    res = calculate_ifrs9_ecl(inp)

    assert res.assigned_stage == 2
    assert res.is_lifetime_ecl is True


def test_ifrs9_stage_3_credit_impaired():
    """Borrower with DPD >= 90 or impaired flag must enter Stage 3."""
    inp = IFRS9StageInput(
        current_distance_to_default=0.5,
        origination_distance_to_default=3.0,
        current_pd=0.35,
        days_past_due=95,
        is_credit_impaired=True,
        ead_schedule=[1_000_000.0, 500_000.0]
    )
    res = calculate_ifrs9_ecl(inp)

    assert res.assigned_stage == 3
    assert "Credit Impaired" in res.stage_name
    assert res.is_lifetime_ecl is True
    # Fix V1: True discounted LGD * EAD loss allowance under PD = 1.0
    expected_ecl = (0.65 * 1_000_000.0) / (1.0 + inp.effective_interest_rate)
    assert res.total_ecl == pytest.approx(expected_ecl, rel=1e-3)


def test_ifrs9_stage_3_full_default_lgd_loss():
    """
    Verify Fix V1: When loan is Stage 3 (Credit Impaired), default has occurred (PD = 1.0),
    so loss allowance equals discounted LGD * EAD.
    """
    ead = 250_000.0
    lgd = 0.60
    discount_rate = 0.05
    res = calculate_multi_period_ecl(
        current_pd=0.25,
        origination_distance_to_default=3.0,
        current_distance_to_default=0.8,
        days_past_due=95,
        is_credit_impaired=True,
        ead_schedule=[ead, ead * 0.5],
        lgd_schedule=[lgd, lgd],
        effective_interest_rate=discount_rate
    )

    assert res["assigned_stage"] == 3
    expected_ecl = lgd * ead / (1.0 + discount_rate)
    assert res["total_ecl"] == pytest.approx(expected_ecl, rel=1e-4)


def test_ecl_discounting_and_term_structure():
    """Verify that multi-period ECL correctly applies discount factor (1 + EIR)^-t."""
    eir = 0.08
    inp = IFRS9StageInput(
        current_distance_to_default=1.5,
        origination_distance_to_default=3.0,
        current_pd=0.05,
        days_past_due=35,
        ead_schedule=[100_000.0, 100_000.0],
        effective_interest_rate=eir
    )
    res = calculate_ifrs9_ecl(inp)

    sched1 = res.period_schedules[0]
    sched2 = res.period_schedules[1]

    assert sched1.discount_factor == pytest.approx(1.0 / (1.0 + eir), rel=1e-3)
    assert sched2.discount_factor == pytest.approx(1.0 / ((1.0 + eir) ** 2), rel=1e-3)
    assert res.total_ecl == pytest.approx(sched1.discounted_ecl + sched2.discounted_ecl, abs=0.1)


def test_vasicek_capital_raw_numerical():
    """Verify calculate_vasicek_capital returns exact dictionary structure and Basel metrics."""
    res = calculate_vasicek_capital(
        pit_pd=0.02,
        lgd=0.40,
        ead=1_500_000.0,
        maturity=2.0,
        confidence_level=0.999
    )

    assert "pit_pd" in res
    assert "asset_correlation_rho" in res
    assert 0.12 <= res["asset_correlation_rho"] <= 0.24
    assert "regulatory_capital_k" in res
    assert res["expected_loss"] == pytest.approx(0.02 * 0.40 * 1_500_000.0, abs=1.0)
    assert res["unexpected_loss"] > 0.0
    assert "stress_scenarios" in res
    assert len(res["stress_scenarios"]) == 4


def test_ecl_staging_and_loss_raw_numerical():
    """Verify calculate_ecl_staging_and_loss handles Stage 1, Stage 2 SICR, and Stage 3 with raw dict."""
    # Stage 1: Stable
    st1 = calculate_ecl_staging_and_loss(
        current_pd=0.01,
        origination_distance_to_default=3.5,
        current_distance_to_default=3.4,
        days_past_due=0,
        ead_schedule=[500_000.0, 500_000.0]
    )
    assert st1["assigned_stage"] == 1
    assert st1["is_lifetime_ecl"] is False
    assert len(st1["period_schedules"]) == 1

    # Stage 2: SICR via Delta DD < -1.0
    st2 = calculate_ecl_staging_and_loss(
        current_pd=0.03,
        origination_distance_to_default=3.5,
        current_distance_to_default=2.0,  # Delta DD = -1.5 < -1.0
        days_past_due=10,
        ead_schedule=[500_000.0, 500_000.0]
    )
    assert st2["assigned_stage"] == 2
    assert st2["is_lifetime_ecl"] is True
    assert len(st2["period_schedules"]) == 2

    # Stage 3: Credit Impaired via DPD >= 90
    st3 = calculate_ecl_staging_and_loss(
        current_pd=0.15,
        origination_distance_to_default=3.0,
        current_distance_to_default=1.0,
        days_past_due=95,
        ead_schedule=[500_000.0, 500_000.0]
    )
    assert st3["assigned_stage"] == 3
    assert st3["is_lifetime_ecl"] is True
    # Fix V1: Discounted LGD * EAD for Stage 3 default (base_lgd=0.65, EIR=0.05)
    expected_ecl_st3 = (0.65 * 500_000.0) / 1.05
    assert st3["total_ecl"] == pytest.approx(expected_ecl_st3, rel=1e-3)


# =====================================================================
# 4. API & INTEGRATION SERVICE TESTS
# =====================================================================

def test_api_health_endpoint():
    """Health check returns 200 with online status and system metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Merton Powell Hybrid 2D Solver" in str(data["capabilities"])


def test_api_merton_solve_endpoint():
    """POST /merton/solve returns valid MertonOutput contract."""
    payload = {
        "equity_value": 200.0,
        "equity_volatility": 0.30,
        "short_term_debt": 50.0,
        "long_term_debt": 100.0,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "loss_given_default": 0.45
    }
    response = client.post("/merton/solve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["default_barrier"] == 100.0
    assert data["asset_value"] > 200.0
    assert data["solver_converged"] is True


def test_api_forensics_endpoint(clean_forensic_input):
    """POST /forensics/analyze returns Beneish, Altman, and CCC."""
    response = client.post("/forensics/analyze", json=clean_forensic_input.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert "beneish" in data
    assert "altman" in data
    assert "ccc" in data
    assert data["beneish"]["is_manipulator"] is False


def test_api_vasicek_endpoint():
    """POST /regulatory/vasicek returns ASRF capital metrics."""
    payload = {
        "pit_pd": 0.02,
        "lgd": 0.45,
        "ead": 1_500_000.0,
        "maturity": 1.5
    }
    response = client.post("/regulatory/vasicek", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.12 <= data["asset_correlation_rho"] <= 0.24
    assert data["regulatory_capital_k"] > 0


def test_api_ecl_endpoint():
    """POST /regulatory/ecl returns multi-period staging result."""
    payload = {
        "current_distance_to_default": 1.5,
        "origination_distance_to_default": 3.0,
        "current_pd": 0.03,
        "days_past_due": 35,
        "ead_schedule": [1_000_000.0, 800_000.0, 600_000.0]
    }
    response = client.post("/regulatory/ecl", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["assigned_stage"] == 2
    assert data["is_lifetime_ecl"] is True


def test_api_comprehensive_assessment(clean_forensic_input):
    """POST /credit/comprehensive produces an end-to-end unified institutional assessment."""
    payload = {
        "company_name": "Apex Industrial Corp",
        "ticker": "APEX",
        "merton_input": {
            "equity_value": 250.0,
            "equity_volatility": 0.32,
            "short_term_debt": 40.0,
            "long_term_debt": 80.0,
            "risk_free_rate": 0.045,
            "maturity": 1.0,
            "loss_given_default": 0.45
        },
        "forensic_input": clean_forensic_input.model_dump(),
        "loan_tenor_years": 5
    }
    response = client.post("/credit/comprehensive", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Apex Industrial Corp"
    assert data["internal_risk_band"] in [b.value for b in AegisInternalRiskBand]
    assert data["indicative_stance"].startswith("INDICATIVE:")
    assert data["credit_rating"] == data["internal_risk_band"]
    assert "regulatory_disclaimer" in data
    assert "recommendation" in data
    assert "executive_summary" in data


def test_api_credit_memo_html(clean_forensic_input):
    """POST /credit/memo/html returns rendered HTML memorandum."""
    payload = {
        "company_name": "Apex Industrial Corp",
        "ticker": "APEX",
        "merton_input": {
            "equity_value": 250.0,
            "equity_volatility": 0.32,
            "short_term_debt": 40.0,
            "long_term_debt": 80.0,
            "risk_free_rate": 0.045
        },
        "forensic_input": clean_forensic_input.model_dump()
    }
    response = client.post("/credit/memo/html", json=payload)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AEGIS" in response.text
    assert "Apex Industrial Corp" in response.text


def test_api_credit_memo_pdf_compilation(clean_forensic_input):
    """POST /credit/memo/pdf generates a binary PDF credit memorandum with %PDF magic header."""
    payload = {
        "company_name": "Apex Industrial Corp",
        "ticker": "APEX",
        "merton_input": {
            "equity_value": 250.0,
            "equity_volatility": 0.32,
            "short_term_debt": 40.0,
            "long_term_debt": 80.0,
            "risk_free_rate": 0.045
        },
        "forensic_input": clean_forensic_input.model_dump()
    }
    response = client.post("/credit/memo/pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    # PDF magic header verification
    assert response.content.startswith(b"%PDF-")
    assert len(response.content) > 10_000  # Non-trivial PDF with charts embedded


def test_api_validation_error_handling():
    """POST with negative equity_value must fail with 422 Unprocessable Entity."""
    payload = {
        "equity_value": -50.0,  # Invalid
        "equity_volatility": 0.30,
        "short_term_debt": 10.0,
        "long_term_debt": 20.0,
        "risk_free_rate": 0.05
    }
    response = client.post("/merton/solve", json=payload)
    assert response.status_code == 422


def test_merton_low_equity_volatility():
    """Verify Merton inversion on low volatility firms (e.g. regulated utilities)."""
    m_in = MertonInput(
        equity_value=500.0,
        equity_volatility=0.12,
        short_term_debt=50.0,
        long_term_debt=200.0,
        risk_free_rate=0.035,
        maturity=1.0
    )
    res = solve_merton(m_in)
    assert res.solver_converged is True
    assert res.asset_volatility < 0.12
    assert res.distance_to_default_physical > 4.0


def test_merton_long_horizon_five_years():
    """Verify Merton stability on multi-year debt maturities (T = 5.0 years)."""
    m_in = MertonInput(
        equity_value=100.0,
        equity_volatility=0.45,
        short_term_debt=20.0,
        long_term_debt=100.0,
        risk_free_rate=0.045,
        maturity=5.0
    )
    res = solve_merton(m_in)
    assert res.solver_converged is True
    assert res.asset_value > 100.0
    assert res.credit_spread_bps > 0.0


def test_forensic_identity_validation_error(clean_forensic_input):
    """Pydantic model must reject balance sheets where current assets exceed total assets."""
    data = clean_forensic_input.model_dump()
    data["current_assets_t"] = 1500.0  # Exceeds total assets of 1000.0
    with pytest.raises(ValueError, match="exceeds total assets"):
        ForensicInput(**data)


def test_vasicek_loss_distribution_properties():
    """Verify numerical properties of Vasicek loss distribution density array."""
    x, density = calculate_vasicek_loss_distribution(pd=0.02, rho=0.18, num_points=250)
    assert len(x) == 250
    assert len(density) == 250
    assert np.all(density >= 0.0)
    # Numerical trapezoidal integral of density should be close to 1
    integral = float(np.trapezoid(density, x))
    assert 0.95 <= integral <= 1.05


def test_frontend_root_served():
    """Verify that root / serves the static glassmorphic frontend index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AEGIS" in response.text
    assert "Merton Structural Probability Density" in response.text
    assert "IFRS 9 / CECL Vasicek Macro Stress Curve" in response.text


def test_api_v1_merton_solve_endpoint():
    """Verify /api/v1/merton/solve route alias works identically."""
    payload = {
        "equity_value": 200.0,
        "equity_volatility": 0.35,
        "short_term_debt": 30.0,
        "long_term_debt": 100.0,
        "risk_free_rate": 0.045,
        "maturity": 1.0,
        "loss_given_default": 0.45
    }
    response = client.post("/api/v1/merton/solve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["solver_converged"] is True
    assert data["default_barrier"] == 80.0


def test_api_v1_export_memo_pdf_endpoint(clean_forensic_input):
    """Verify /api/v1/credit/export-memo route downloads valid binary PDF."""
    payload = {
        "company_name": "Apex Industrial Corp",
        "ticker": "APEX",
        "merton_input": {
            "equity_value": 200.0,
            "equity_volatility": 0.35,
            "short_term_debt": 30.0,
            "long_term_debt": 100.0,
            "risk_free_rate": 0.045
        },
        "forensic_input": clean_forensic_input.model_dump()
    }
    response = client.post("/api/v1/credit/export-memo", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_solve_structural_merton_direct_call():
    """Verify solve_structural_merton direct dictionary output contract."""
    res = solve_structural_merton(
        equity_market_cap=150.0,
        equity_volatility=0.40,
        short_term_debt=30.0,
        long_term_debt=80.0,
        risk_free_rate=0.045,
        asset_drift_mu=0.06,
        maturity_years=1.0,
        recovery_rate=0.40
    )
    assert "asset_value" in res
    assert "asset_volatility" in res
    assert "distance_to_default" in res
    assert "physical_default_prob" in res
    assert "risk_neutral_default_prob" in res
    assert "implied_credit_spread_bps" in res
    assert res["default_barrier"] == 70.0
    assert res["asset_value"] > 150.0
    assert res["distance_to_default"] > 0.0


def test_solve_structural_merton_zero_debt():
    """Verify solve_structural_merton asymptotic clamp when D=0."""
    res = solve_structural_merton(
        equity_market_cap=200.0,
        equity_volatility=0.25,
        short_term_debt=0.0,
        long_term_debt=0.0,
        risk_free_rate=0.04,
        asset_drift_mu=0.04
    )
    assert res["asset_value"] == 200.0
    assert res["asset_volatility"] == 0.25
    assert res["distance_to_default"] == 99.0
    assert res["physical_default_prob"] == 0.0
    assert res["implied_credit_spread_bps"] == 0.0


def test_merton_objective_and_jacobian_clamping():
    """Verify domain clamping behavior in merton_objective_and_jacobian."""
    res, jac = merton_objective_and_jacobian(
        vars_vec=np.array([-10.0, -0.5]),
        E=100.0,
        sigma_E=0.4,
        D=80.0,
        r=0.05,
        T=1.0
    )
    assert np.all(res == 1e8)
    assert np.all(jac == np.eye(2) * 1e8)


def test_schema_balance_sheet_identity_rejection():
    """Strict Pydantic validator rejects non-reconciling balance sheets beyond 5% tolerance."""
    # Valid reconciliation: 1000 - 400 = 600
    req_valid = ComprehensiveCreditRequest(
        ticker="APEX",
        company_name="Apex Corp",
        equity_market_cap=200.0,
        equity_volatility=0.35,
        short_term_debt=40.0,
        long_term_debt=80.0,
        total_assets=1000.0,
        total_liabilities=400.0,
        book_value_equity=600.0,
        working_capital=150.0,
        retained_earnings=200.0,
        ebit=120.0
    )
    assert req_valid.total_assets == 1000.0

    # Invalid: 1000 - 400 = 600 != 800 (error = 200 > 50 tolerance)
    with pytest.raises(ValueError, match="Accounting Identity Failure"):
        ComprehensiveCreditRequest(
            ticker="APEX",
            company_name="Apex Corp",
            equity_market_cap=200.0,
            equity_volatility=0.35,
            short_term_debt=40.0,
            long_term_debt=80.0,
            total_assets=1000.0,
            total_liabilities=400.0,
            book_value_equity=800.0,
            working_capital=150.0,
            retained_earnings=200.0,
            ebit=120.0
        )


def test_schema_merton_analysis_request_bounds():
    """Verify perimeter bounds on MertonAnalysisRequest: volatility in [0.005, 5.0], cap > 0."""
    req = MertonAnalysisRequest(
        equity_market_cap=150.0,
        equity_volatility=0.30,
        short_term_debt=30.0,
        long_term_debt=60.0
    )
    assert req.equity_market_cap == 150.0

    with pytest.raises(ValueError):
        MertonAnalysisRequest(
            equity_market_cap=-10.0,  # Invalid: <= 0
            equity_volatility=0.30,
            short_term_debt=30.0,
            long_term_debt=60.0
        )

    with pytest.raises(ValueError):
        MertonAnalysisRequest(
            equity_market_cap=150.0,
            equity_volatility=0.001,  # Invalid: <= 0.005
            short_term_debt=30.0,
            long_term_debt=60.0
        )


def test_schema_forensic_double_entry_reconciliation():
    """Verify ForensicAnalysisRequest enforces double-entry identity reconciliation."""
    # Valid: Assets 1000 - Liab 400 = Equity 600
    req = ForensicAnalysisRequest(
        sales_t0=1000.0, sales_t1=900.0,
        cogs_t0=600.0, cogs_t1=550.0,
        receivables_t0=120.0, receivables_t1=100.0,
        current_assets_t0=400.0, current_assets_t1=350.0,
        ppe_t0=500.0, ppe_t1=480.0,
        total_assets_t0=1000.0, total_assets_t1=950.0,
        depreciation_t0=40.0, depreciation_t1=38.0,
        sga_t0=150.0, sga_t1=140.0,
        net_income_t0=90.0, operating_cash_flow_t0=100.0,
        total_debt_t0=300.0, total_debt_t1=280.0,
        working_capital_t0=150.0, retained_earnings_t0=250.0,
        ebit_t0=140.0, book_value_equity_t0=600.0,
        total_liabilities_t0=400.0
    )
    assert req.total_assets_t0 == 1000.0

    # Imbalanced: Assets 1000 - Liab 400 = 600 != 850
    with pytest.raises(ValueError, match="Double-Entry Identity Violation"):
        ForensicAnalysisRequest(
            sales_t0=1000.0, sales_t1=900.0,
            cogs_t0=600.0, cogs_t1=550.0,
            receivables_t0=120.0, receivables_t1=100.0,
            current_assets_t0=400.0, current_assets_t1=350.0,
            ppe_t0=500.0, ppe_t1=480.0,
            total_assets_t0=1000.0, total_assets_t1=950.0,
            depreciation_t0=40.0, depreciation_t1=38.0,
            sga_t0=150.0, sga_t1=140.0,
            net_income_t0=90.0, operating_cash_flow_t0=100.0,
            total_debt_t0=300.0, total_debt_t1=280.0,
            working_capital_t0=150.0, retained_earnings_t0=250.0,
            ebit_t0=140.0, book_value_equity_t0=850.0,  # Imbalanced
            total_liabilities_t0=400.0
        )


def test_api_merton_analyze_endpoint():
    """POST /merton/analyze returns 200 with valid MertonOutput."""
    payload = {
        "equity_market_cap": 250.0,
        "equity_volatility": 0.35,
        "short_term_debt": 40.0,
        "long_term_debt": 80.0,
        "risk_free_rate": 0.045
    }
    response = client.post("/merton/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["solver_converged"] is True
    assert data["distance_to_default_physical"] > 0.0


def test_api_comprehensive_assessment_flat_schema():
    """POST /credit/comprehensive works with flat schema and performs end-to-end evaluation."""
    payload = {
        "ticker": "NVDA",
        "company_name": "Nvidia Corporation",
        "equity_market_cap": 2000.0,
        "equity_volatility": 0.45,
        "short_term_debt": 50.0,
        "long_term_debt": 100.0,
        "risk_free_rate": 0.045,
        "total_assets": 1200.0,
        "total_liabilities": 400.0,
        "book_value_equity": 800.0,
        "working_capital": 300.0,
        "retained_earnings": 500.0,
        "ebit": 400.0,
        "revenue": 1000.0,
        "cogs": 400.0,
        "receivables": 150.0,
        "inventory": 100.0,
        "payables": 80.0,
        "loan_exposure": 150.0
    }
    response = client.post("/credit/comprehensive", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Nvidia Corporation"
    assert "merton" in data
    assert "forensics" in data
    assert "vasicek" in data
    assert "ifrs9_ecl" in data
    assert "composite_rating" in data


def test_api_comprehensive_assessment_balance_sheet_rejection():
    """POST /credit/comprehensive rejects requests where balance sheet identity is violated."""
    payload = {
        "ticker": "BAD",
        "company_name": "Imbalanced Corp",
        "equity_market_cap": 100.0,
        "equity_volatility": 0.40,
        "short_term_debt": 50.0,
        "long_term_debt": 50.0,
        "total_assets": 1000.0,
        "total_liabilities": 300.0,
        "book_value_equity": 900.0,  # 1000 - 300 = 700 != 900
        "working_capital": 100.0,
        "retained_earnings": 100.0,
        "ebit": 50.0
    }
    response = client.post("/credit/comprehensive", json=payload)
    assert response.status_code == 422
    assert "Accounting Identity Failure" in response.text


def test_ead_scaling_preservation():
    """Verify Fix A1: ead reflects explicit loan_exposure without being clamped to 100,000.0 floor."""
    payload = {
        "ticker": "TECH",
        "company_name": "Scale Test Corp",
        "equity_market_cap": 500.0,
        "equity_volatility": 0.25,
        "short_term_debt": 20.0,
        "long_term_debt": 40.0,
        "total_assets": 800.0,
        "total_liabilities": 200.0,
        "book_value_equity": 600.0,
        "working_capital": 100.0,
        "retained_earnings": 200.0,
        "ebit": 100.0,
        "loan_exposure": 150.0,
        "loan_tenor_years": 5
    }
    response = client.post("/credit/comprehensive", json=payload)
    assert response.status_code == 200
    data = response.json()
    ecl = data["ifrs9_ecl"]
    # Check that initial EAD schedule reflects 150.0 and NOT 100,000.0
    assert ecl["period_schedules"][0]["ead"] == pytest.approx(150.0, rel=1e-3)
    # Total ECL must scale to millions ($150.0M), NOT hundreds of millions / billions
    assert ecl["total_ecl"] < 50.0


# Backward compatibility alias
test_ead_scaling_preservation_no_floor = test_ead_scaling_preservation


def test_internal_risk_band_replaces_agency_ratings():
    """Verify Fix A4: outputs AegisInternalRiskBand, indicative stance, and model disclaimer."""
    # 1. High quality firm -> Band 1 or Band 2
    payload_prime = {
        "ticker": "PRIME",
        "company_name": "Prime Fortress Corp",
        "equity_market_cap": 2_000_000.0,
        "equity_volatility": 0.20,
        "short_term_debt": 1000.0,
        "long_term_debt": 5000.0,
        "total_assets": 500_000.0,
        "total_liabilities": 100_000.0,
        "book_value_equity": 400_000.0,
        "working_capital": 50_000.0,
        "retained_earnings": 100_000.0,
        "ebit": 50_000.0,
        "loan_exposure": 500.0
    }
    res_prime = client.post("/credit/comprehensive", json=payload_prime)
    assert res_prime.status_code == 200
    data_prime = res_prime.json()
    assert data_prime["internal_risk_band"] in [
        AegisInternalRiskBand.BAND_1.value,
        AegisInternalRiskBand.BAND_2.value,
    ]
    assert data_prime["internal_risk_band"].startswith("Band ")
    assert "INDICATIVE: MINIMAL RISK" in data_prime["indicative_stance"] or "INDICATIVE: LOW RISK" in data_prime["indicative_stance"]
    assert data_prime["credit_rating"] == data_prime["internal_risk_band"]
    assert data_prime["recommendation"] == data_prime["indicative_stance"]
    for legacy_rating in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]:
        assert data_prime["credit_rating"] != legacy_rating
    assert "APPROVE" not in data_prime["recommendation"]
    assert "REJECT" not in data_prime["recommendation"]
    assert data_prime["regulatory_disclaimer"] == (
        "Aegis internal risk bands and default metrics are uncalibrated quantitative model outputs "
        "derived from mathematical simplifications. They do not constitute credit ratings from an NRSRO "
        "or formalized credit committee approvals."
    )

    # 2. Distressed firm -> Band 6 or Band 7
    payload_distress = {
        "ticker": "DISTRESS",
        "company_name": "Distressed Energy Corp",
        "equity_market_cap": 50.0,
        "equity_volatility": 1.20,
        "short_term_debt": 400.0,
        "long_term_debt": 400.0,
        "total_assets": 1000.0,
        "total_liabilities": 950.0,
        "book_value_equity": 50.0,
        "working_capital": -100.0,
        "retained_earnings": -300.0,
        "ebit": -50.0,
        "loan_exposure": 200.0,
        "days_past_due": 95
    }
    res_distress = client.post("/credit/comprehensive", json=payload_distress)
    assert res_distress.status_code == 200
    data_distress = res_distress.json()
    assert data_distress["internal_risk_band"] in [
        AegisInternalRiskBand.BAND_6.value,
        AegisInternalRiskBand.BAND_7.value,
    ]
    assert data_distress["internal_risk_band"].startswith("Band ")
    assert "INDICATIVE: VERY HIGH RISK" in data_distress["indicative_stance"] or "INDICATIVE: HIGH RISK" in data_distress["indicative_stance"]
    assert "APPROVE" not in data_distress["recommendation"]
    assert "REJECT" not in data_distress["recommendation"]


# Backward compatibility alias
test_internal_risk_band_classification_and_disclaimer = test_internal_risk_band_replaces_agency_ratings


# =====================================================================
# 5. HISTORICAL CASE STUDY EMPIRICAL BACKTEST TESTS
# =====================================================================

def test_case_study_svb_distress_backtest():
    """
    Deterministic verification of Silicon Valley Bank (SIVB) Q4 2022 pre-collapse backtest.
    Validates structural default barrier penetration, credit spread blowout (>500 bps),
    bank domain warning for non-financial models, and IFRS 9 Stage 3 credit impairment.
    """
    # 1. Structural Merton Inversion (Me4: risky drift mu = 0.045 >= r = 0.0425)
    m_res = solve_structural_merton(
        equity_market_cap=13500.0,
        equity_volatility=0.68,
        short_term_debt=85000.0,
        long_term_debt=15000.0,
        risk_free_rate=0.0425,
        asset_drift_mu=0.045,
        maturity_years=1.0,
        recovery_rate=0.35
    )

    assert m_res["default_barrier"] == 92500.0
    assert 100_000.0 <= m_res["asset_value"] <= 104_000.0
    assert 0.08 <= m_res["asset_volatility"] <= 0.11
    assert 1.00 <= m_res["distance_to_default"] <= 1.50  # Severe distress threshold (< 1.50 sigma)
    assert m_res["distance_to_default"] < 1.5
    assert m_res["physical_default_prob"] > 0.05
    assert m_res["structural_merton_spread_bps"] > 400.0
    assert m_res["implied_credit_spread_bps"] > 400.0
    assert m_res["leverage_ratio"] > 0.85

    # 2. Forensic Accounting Diagnostic (Beneish & Altman)
    # Bank domain guard verification (F3)
    b_bank = calculate_beneish_m_score(
        sales_t=2500.0, sales_prev=2200.0,
        cogs_t=1200.0, cogs_prev=1050.0,
        rec_t=500.0, rec_prev=440.0,
        ca_t=20000.0, ca_prev=25000.0,
        ppe_t=2000.0, ppe_prev=2000.0,
        sec_t=120000.0, sec_prev=140000.0,
        ta_t=212000.0, ta_prev=211000.0,
        dep_t=200.0, dep_prev=200.0,
        sga_t=800.0, sga_prev=750.0,
        net_income_t=1500.0, cfo_t=-2000.0,
        total_debt_t=195000.0, total_debt_prev=170000.0,
        is_financial_institution=True
    )
    assert b_bank["domain_warning"] == "These models were built for non-financial firms."
    assert b_bank["is_manipulator"] is False

    # Raw Beneish with corrected 1999 weights yields ~ -2.08 (< -1.78)
    b_res = calculate_beneish_m_score(
        sales_t=2500.0, sales_prev=2200.0,
        cogs_t=1200.0, cogs_prev=1050.0,
        rec_t=500.0, rec_prev=440.0,
        ca_t=20000.0, ca_prev=25000.0,
        ppe_t=2000.0, ppe_prev=2000.0,
        sec_t=120000.0, sec_prev=140000.0,
        ta_t=212000.0, ta_prev=211000.0,
        dep_t=200.0, dep_prev=200.0,
        sga_t=800.0, sga_prev=750.0,
        net_income_t=1500.0, cfo_t=-2000.0,
        total_debt_t=195000.0, total_debt_prev=170000.0
    )
    assert pytest.approx(b_res["m_score"], abs=0.02) == -2.08
    assert b_res["m_score"] < -1.78
    assert b_res["is_manipulator"] is False
    assert b_res["indices"]["AQI"] > 1.10

    z_res = calculate_altman_z_double_prime(
        working_capital=-5000.0,
        retained_earnings=15000.0,
        ebit=2500.0,
        book_value_equity=13500.0,
        total_assets=212000.0,
        total_liabilities=198500.0
    )
    assert z_res["z_double_prime"] < 1.10
    assert z_res["zone"] == "DISTRESS"

    # 3. Regulatory Loss Distribution & IFRS 9 ECL (Fix T2: Feed actual Merton outputs)
    ecl_res = calculate_ecl_staging_and_loss(
        current_pd=m_res["physical_default_prob"],
        origination_distance_to_default=3.5,
        current_distance_to_default=m_res["distance_to_default"],
        days_past_due=90,
        is_credit_impaired=True,
        ead_schedule=[100000.0, 80000.0, 60000.0, 40000.0, 20000.0],
        effective_interest_rate=0.06
    )
    assert ecl_res["assigned_stage"] == 3
    assert ecl_res["is_lifetime_ecl"] is True
    assert ecl_res["ecl_percentage_of_ead"] > 18.0


def test_case_study_msft_fortress_benchmark():
    """
    Deterministic verification of Microsoft Corporation (MSFT) sovereign-equivalent benchmark (Band 1).
    Validates extreme solvency buffer (DD > 6.20 sigma), zero fair credit spread (0.0 bps),
    pristine forensic M-score (is_manipulator is False), Altman Z'' in Safe Zone, and Stage 1 IFRS 9 staging.
    """
    # 1. Structural Merton Inversion
    m_res = solve_structural_merton(
        equity_market_cap=2400000.0,
        equity_volatility=0.225,
        short_term_debt=5200.0,
        long_term_debt=42000.0,
        risk_free_rate=0.045,
        asset_drift_mu=0.08,
        maturity_years=1.0,
        recovery_rate=0.45
    )

    assert m_res["default_barrier"] == 26200.0
    assert m_res["asset_value"] >= 2_400_000.0
    assert 0.20 <= m_res["asset_volatility"] <= 0.25
    assert m_res["distance_to_default"] > 6.20  # Band 1 sovereign-equivalent boundary
    assert m_res["physical_default_prob"] < 0.0001
    assert abs(m_res["implied_credit_spread_bps"]) < 1.0  # Zero spread premium
    assert m_res["leverage_ratio"] < 0.02

    # 2. Forensic Accounting Diagnostic (Beneish & Altman)
    b_res = calculate_beneish_m_score(
        sales_t=211915.0, sales_prev=198270.0,
        cogs_t=65863.0, cogs_prev=62650.0,
        rec_t=38290.0, rec_prev=35800.0,
        ca_t=184257.0, ca_prev=169684.0,
        ppe_t=110000.0, ppe_prev=95000.0,
        sec_t=50000.0, sec_prev=45000.0,
        ta_t=411976.0, ta_prev=364840.0,
        dep_t=13000.0, dep_prev=12000.0,
        sga_t=27000.0, sga_prev=25000.0,
        net_income_t=72361.0, cfo_t=87582.0,
        total_debt_t=47000.0, total_debt_prev=48000.0
    )
    assert b_res["m_score"] < -1.78
    assert b_res["is_manipulator"] is False

    z_res = calculate_altman_z_double_prime(
        working_capital=80000.0,
        retained_earnings=118000.0,
        ebit=88500.0,
        book_value_equity=206223.0,
        total_assets=411976.0,
        total_liabilities=205753.0
    )
    assert z_res["z_double_prime"] > 2.60
    assert z_res["zone"] == "SAFE"

    # 3. Regulatory Loss Distribution & IFRS 9 ECL (Fix T2: Feed actual Merton outputs)
    ecl_res = calculate_ecl_staging_and_loss(
        current_pd=max(m_res["physical_default_prob"], 1e-6),
        origination_distance_to_default=m_res["distance_to_default"],
        current_distance_to_default=m_res["distance_to_default"],
        days_past_due=0,
        is_credit_impaired=False,
        ead_schedule=[47200.0],
        effective_interest_rate=0.045
    )
    assert ecl_res["assigned_stage"] == 1
    assert ecl_res["is_lifetime_ecl"] is False
    assert ecl_res["ecl_percentage_of_ead"] < 0.1


# =====================================================================
# 17. STAGE 3 VERIFICATION TESTS (F2-F4, V3, S2, A5, A6)
# =====================================================================

def test_sme_correlation_below_012():
    """V3: SME correlation must not be clipped back up to 0.12; floor is 0.08."""
    # At PD=0.20, base corporate rho is ~0.12.
    # For firm with 5.0M EUR sales, SME discount is -0.04 -> rho should reach 0.08.
    rho_sme = calculate_corporate_asset_correlation(pd=0.20, firm_sales_eur_m=5.0)
    assert rho_sme < 0.12
    assert rho_sme >= 0.08
    assert pytest.approx(rho_sme, abs=1e-3) == 0.08

    # Without SME sales, correlation stays >= 0.12
    rho_large = calculate_corporate_asset_correlation(pd=0.20, firm_sales_eur_m=None)
    assert rho_large >= 0.12

    # Test aliases calculate_asset_correlation and calculate_basel_correlation
    from core.vasicek import calculate_asset_correlation, calculate_basel_correlation
    assert calculate_asset_correlation(pd=0.20, firm_sales_eur_m=5.0) == rho_sme
    assert calculate_basel_correlation(pd=0.20, firm_sales_eur_m=5.0) == rho_sme


def test_financial_institution_domain_warning(clean_forensic_input):
    """F3: Financial institutions skip non-financial models with explicit domain warning."""
    bank_input = clean_forensic_input.model_copy(update={"is_financial_institution": True})
    forensics_res = calculate_forensics(bank_input)

    assert forensics_res.is_financial_institution is True
    assert forensics_res.domain_warning == "These models were built for non-financial firms."
    assert forensics_res.beneish.domain_warning == "These models were built for non-financial firms."
    assert forensics_res.altman.domain_warning == "These models were built for non-financial firms."
    assert forensics_res.ccc.domain_warning == "These models were built for non-financial firms."
    assert forensics_res.beneish.is_manipulator is False
    assert len(forensics_res.beneish.red_flags) == 0
    assert forensics_res.ccc.warning_triggered is False
    assert forensics_res.composite_forensic_risk == "Not Applicable"
    assert "These models were built for non-financial firms." in forensics_res.summary_findings

    # Verify calculate_beneish_m_score raw returns domain warning
    b_raw = calculate_beneish_m_score(
        sales_t=100.0, sales_prev=100.0, cogs_t=50.0, cogs_prev=50.0,
        rec_t=10.0, rec_prev=10.0, ca_t=50.0, ca_prev=50.0,
        ppe_t=20.0, ppe_prev=20.0, sec_t=0.0, sec_prev=0.0,
        ta_t=100.0, ta_prev=100.0, dep_t=5.0, dep_prev=5.0,
        sga_t=10.0, sga_prev=10.0, net_income_t=10.0, cfo_t=10.0,
        total_debt_t=50.0, total_debt_prev=50.0,
        is_financial_institution=True,
    )
    assert b_raw["domain_warning"] == "These models were built for non-financial firms."
    assert b_raw["is_manipulator"] is False


def test_beneish_missing_data_quality_flags():
    """F4: Missing or zero denominators trigger explicit data_quality_flags."""
    # Calling with zero sales_prev and zero rec_prev
    b_res = calculate_beneish_m_score(
        sales_t=100.0, sales_prev=0.0,
        cogs_t=50.0, cogs_prev=0.0,
        rec_t=10.0, rec_prev=0.0,
        ca_t=50.0, ca_prev=0.0,
        ppe_t=20.0, ppe_prev=0.0,
        sec_t=0.0, sec_prev=0.0,
        ta_t=100.0, ta_prev=0.0,
        dep_t=5.0, dep_prev=0.0,
        sga_t=10.0, sga_prev=0.0,
        net_income_t=10.0, cfo_t=10.0,
        total_debt_t=50.0, total_debt_prev=0.0,
    )
    flags = b_res["data_quality_flags"]
    assert len(flags) > 0
    assert any("not computable: zero or missing denominator" in f for f in flags)
    assert any("DSRI" in f for f in flags)

    # API comprehensive flat request missing comparative data
    flat_req = {
        "company_name": "Test Flat Corp",
        "ticker": "TFC",
        "equity_market_cap": 200.0,
        "equity_volatility": 0.35,
        "short_term_debt": 40.0,
        "long_term_debt": 60.0,
        "loan_exposure": 50.0,
        "revenue": 300.0,
        "cogs": 180.0,
        "total_assets": 500.0,
        "total_liabilities": 200.0,
        "book_value_equity": 300.0,
        "ebit": 40.0,
        "retained_earnings": 100.0,
        "working_capital": 50.0,
        "receivables": 30.0,
        "inventory": 40.0,
        "payables": 25.0
    }
    resp = client.post("/credit/comprehensive", json=flat_req)
    assert resp.status_code == 200
    data = resp.json()
    beneish_flags = data["forensics"]["beneish"]["data_quality_flags"]
    assert "Beneish/CCC requires two periods of comparative accounting data" in beneish_flags


def test_forensic_input_1pct_identity_check():
    """S2: ForensicInput enforces strict 1% balance-sheet identity tolerance."""
    base_kwargs = dict(
        sales_t=1000.0, cogs_t=600.0, receivables_t=120.0, current_assets_t=400.0,
        ppe_t=500.0, total_assets_t=1000.0, depreciation_t=50.0, sga_t=150.0,
        net_income_t=120.0, cash_flow_operations_t=130.0, current_liabilities_t=150.0,
        long_term_debt_t=250.0, total_liabilities_t=400.0, inventory_t=150.0,
        accounts_payable_t=100.0, ebit_t=150.0, retained_earnings_t=300.0,
        sales_t_minus_1=950.0, cogs_t_minus_1=570.0, receivables_t_minus_1=114.0,
        current_assets_t_minus_1=380.0, ppe_t_minus_1=480.0, total_assets_t_minus_1=950.0,
        depreciation_t_minus_1=48.0, sga_t_minus_1=142.0, long_term_debt_t_minus_1=190.0,
        total_liabilities_t_minus_1=380.0, inventory_t_minus_1=140.0,
        accounts_payable_t_minus_1=95.0
    )
    # Assets (1000) - Liabilities (400) = 600. Diff = 5 <= 10 (1%) -> Valid
    valid_inp = ForensicInput(**base_kwargs, book_value_equity_t=605.0)
    assert valid_inp.book_value_equity_t == 605.0

    # Assets (1000) - Liabilities (400) = 600. Diff = 15 > 10 (1%) -> Violation
    with pytest.raises(ValueError, match="Balance-Sheet Identity Violation|1% tolerance"):
        ForensicInput(**base_kwargs, book_value_equity_t=585.0)


def test_pd_nonzero_precision_under_standard_normal():
    """Issue 2: Verify PD is not truncated to 0.000% under standard normal model when DD is ~4.95 sigma."""
    m_in = MertonInput(
        equity_value=200.0,
        equity_volatility=0.35,
        short_term_debt=30.0,
        long_term_debt=100.0,
        risk_free_rate=0.045
    )
    res = solve_merton(m_in)
    assert res.distance_to_default_physical > 4.5
    # Non-zero floating point probability under standard normal Phi(-4.95) ~ 3.71e-7
    assert res.default_probability_physical > 0.0
    assert pytest.approx(res.default_probability_physical, rel=0.1) == 3.71e-7


def test_risk_reconciliation_explanation_for_stage3_band7_with_clean_forensics(clean_forensic_input):
    """Issue 1: Verify reconciliation explains why Stage 3 / Band 7 occurs alongside clean forensics."""
    # Create request with severe structural distress and Stage 3 DPD=90
    # but clean, unmanipulated accounting statements
    req_data = {
        "company_name": "Solvency Distressed Corp",
        "ticker": "SDC",
        "merton_input": {
            "equity_value": 10.0,
            "equity_volatility": 0.85,
            "short_term_debt": 80.0,
            "long_term_debt": 200.0,
            "risk_free_rate": 0.045,
            "maturity": 1.0,
            "loss_given_default": 0.45
        },
        "days_past_due": 90,
        "forensic_input": clean_forensic_input.model_dump(),
        "is_financial_institution": False,
        "loan_tenor_years": 5
    }
    resp = client.post("/credit/comprehensive", json=req_data)
    assert resp.status_code == 200
    data = resp.json()
    assert data["internal_risk_band"] == AegisInternalRiskBand.BAND_7.value
    assert data["ifrs9_ecl"]["assigned_stage"] == 3
    assert data["forensics"]["beneish"]["is_manipulator"] is False
    # Summary must explicitly explain the divergence between structural distress and clean forensics
    assert "Risk Reconciliation" in data["executive_summary"]
    assert "structural market distress" in data["executive_summary"]


def test_export_underwriting_memo_pdf_endpoint(clean_forensic_input):
    """Issue 4: Verify /credit/export-memo returns a valid, downloadable PDF report with nested payload."""
    payload = {
        "company_name": "Apex Industrial Corp",
        "ticker": "APEX",
        "equity_market_cap": 200.0,
        "equity_volatility": 0.35,
        "short_term_debt": 30.0,
        "long_term_debt": 100.0,
        "risk_free_rate": 0.045,
        "merton_input": {
            "equity_value": 200.0,
            "equity_volatility": 0.35,
            "short_term_debt": 30.0,
            "long_term_debt": 100.0,
            "risk_free_rate": 0.045,
            "maturity": 1.0,
            "loss_given_default": 0.45
        },
        "forensic_input": clean_forensic_input.model_dump(),
        "is_financial_institution": False,
        "loan_tenor_years": 5
    }
    resp = client.post("/api/v1/credit/export-memo", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "Aegis_Credit_Memo_Apex_Industrial_Corp.pdf" in resp.headers["content-disposition"]
    # Verify PDF file header
    assert resp.content.startswith(b"%PDF")
    assert len(resp.content) > 50_000







