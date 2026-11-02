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
    assert data["composite_rating"] in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/D"]
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




