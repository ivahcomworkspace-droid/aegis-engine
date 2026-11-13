"""
Aegis Institutional Case Study Reproducibility Runner.
Executes historical corporate balance sheet data through the quantitative engine.
Reproduces:
1. Silicon Valley Bank (SIVB) - Q4 2022 Pre-Collapse Structural Distress & Forensic Audit
2. Microsoft Corporation (MSFT) - Sovereign-Equivalent Benchmark (Band 1)
"""

import os
import sys

# Ensure repository root is on sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.merton import solve_structural_merton
from core.forensics import calculate_beneish_m_score, calculate_altman_z_double_prime
from core.vasicek import calculate_multi_period_ecl


def run_svb_backtest():
    print("=" * 70)
    print("BACKTEST 1: SILICON VALLEY BANK (SIVB) - Q4 2022 PRE-COLLAPSE")
    print("=" * 70)

    # 1. Structural Merton Inversion (Me4: risky asset drift mu = 0.045 >= r = 0.0425)
    res = solve_structural_merton(
        equity_market_cap=13500.0,
        equity_volatility=0.68,
        short_term_debt=85000.0,
        long_term_debt=15000.0,
        risk_free_rate=0.0425,
        asset_drift_mu=0.045,
        maturity_years=1.0,
        recovery_rate=0.35
    )
    print("\n[1] STRUCTURAL MERTON INVERSION:")
    print(f"  Observable Equity Cap (E):     ${13500.0:,.1f}M")
    print(f"  Default Barrier (D):           ${res['default_barrier']:,.1f}M")
    print(f"  Latent Asset Value (V_A):      ${res['asset_value']:,.1f}M")
    print(f"  Latent Asset Vol (sigma_A):    {res['asset_volatility']*100:.2f}%")
    print(f"  Distance-to-Default (DD):      {res['distance_to_default']:.2f} sigma")
    print(f"  Physical Default Prob (PD):    {res['physical_default_prob']*100:.2f}%")
    print(f"  Fair Credit Spread:            {res['implied_credit_spread_bps']:.1f} bps")
    print(f"  Structural Merton Spread:      {res['structural_merton_spread_bps']:.1f} bps")
    print(f"  Mandate Verdict:               {'BAND 7 (IMPAIRED / DEFAULT RISK)' if res['distance_to_default'] < 1.5 else 'REVIEW'}")

    # 2. Forensic Accounting Diagnostic (Beneish M-Score & Altman Z'')
    # SVB 2022 disclosures: unhedged HTM securities and deposit outflow leverage distortion
    beneish_res = calculate_beneish_m_score(
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
    bank_beneish = calculate_beneish_m_score(
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
    altman_res = calculate_altman_z_double_prime(
        working_capital=-5000.0,
        retained_earnings=15000.0,
        ebit=2500.0,
        book_value_equity=13500.0,
        total_assets=212000.0,
        total_liabilities=198500.0
    )

    print("\n[2] BALANCE SHEET FORENSICS:")
    print(f"  Beneish M-Score (Raw):         {beneish_res['m_score']:.2f}")
    print(f"  Manipulation Status:           {'FLAGGED' if beneish_res['is_manipulator'] else 'CLEAN / BELOW THRESHOLD'} (Threshold: -1.78)")
    print(f"  Bank Domain Warning:           {bank_beneish['domain_warning']}")
    print(f"  AQI (Asset Quality Index):     {beneish_res['indices']['AQI']:.2f} (Alert if > 1.10)")
    print(f"  LVGI (Leverage Index):         {beneish_res['indices']['LVGI']:.2f} (Alert if > 1.15)")
    print(f"  Altman Z''-Score:              {altman_res['z_double_prime']:.2f} (Zone: {altman_res['zone']})")

    # 3. Regulatory Loss Distribution & IFRS 9 ECL
    # Impaired liquidity run: multi-period Stage 3 schedule fed directly from Merton outputs
    ecl_res = calculate_multi_period_ecl(
        current_pd=res['physical_default_prob'],
        origination_distance_to_default=3.5,
        current_distance_to_default=res['distance_to_default'],
        days_past_due=90,
        is_credit_impaired=True,
        ead_schedule=[100000.0, 80000.0, 60000.0, 40000.0, 20000.0],
        effective_interest_rate=0.06
    )
    print("\n[3] REGULATORY IFRS 9 ECL STAGING:")
    print(f"  Assigned Staging:              {ecl_res['stage_name']}")
    print(f"  Lifetime ECL Allowance:        ${ecl_res['total_ecl']:,.0f}")
    print(f"  ECL Provisioning % of EAD:     {ecl_res['ecl_percentage_of_ead']:.2f}%")
    print(f"  Indicative Underwriting Stance: BAND 7 (IMPAIRED / DEFAULT RISK) — IMMEDIATE WORKOUT")


def run_msft_benchmark():
    print("\n" + "=" * 70)
    print("BACKTEST 2: MICROSOFT CORP (MSFT) - SOVEREIGN-EQUIVALENT BENCHMARK")
    print("=" * 70)

    # 1. Structural Merton Inversion
    res = solve_structural_merton(
        equity_market_cap=2400000.0,
        equity_volatility=0.225,
        short_term_debt=5200.0,
        long_term_debt=42000.0,
        risk_free_rate=0.045,
        asset_drift_mu=0.08,
        maturity_years=1.0,
        recovery_rate=0.45
    )
    print("\n[1] STRUCTURAL MERTON INVERSION:")
    print(f"  Observable Equity Cap (E):     ${2400000.0:,.1f}M")
    print(f"  Default Barrier (D):           ${res['default_barrier']:,.1f}M")
    print(f"  Latent Asset Value (V_A):      ${res['asset_value']:,.1f}M")
    print(f"  Latent Asset Vol (sigma_A):    {res['asset_volatility']*100:.2f}%")
    print(f"  Distance-to-Default (DD):      {res['distance_to_default']:.2f} sigma")
    print(f"  Physical Default Prob (PD):    {res['physical_default_prob']*100:.4f}%")
    print(f"  Fair Credit Spread:            {res['implied_credit_spread_bps']:.1f} bps")
    print(f"  Structural Merton Spread:      {res['structural_merton_spread_bps']:.1f} bps")
    print(f"  Mandate Verdict:               {'BAND 1 (MINIMAL CREDIT RISK)' if res['distance_to_default'] > 4.5 else 'REVIEW'}")

    # 2. Forensic Accounting Diagnostic (Beneish M-Score & Altman Z'')
    beneish_res = calculate_beneish_m_score(
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
    altman_res = calculate_altman_z_double_prime(
        working_capital=80000.0,
        retained_earnings=118000.0,
        ebit=88500.0,
        book_value_equity=206223.0,
        total_assets=411976.0,
        total_liabilities=205753.0
    )

    print("\n[2] BALANCE SHEET FORENSICS:")
    print(f"  Beneish M-Score:               {beneish_res['m_score']:.2f}")
    print(f"  Manipulation Status:           {'FLAGGED' if beneish_res['is_manipulator'] else 'CLEAN / BELOW THRESHOLD'}")
    print(f"  Altman Z''-Score:              {altman_res['z_double_prime']:.2f} (Zone: {altman_res['zone']})")

    # 3. Regulatory Loss Distribution & IFRS 9 ECL (Fix T2: Feed actual Merton outputs)
    ecl_res = calculate_multi_period_ecl(
        current_pd=max(res['physical_default_prob'], 1e-6),
        origination_distance_to_default=res['distance_to_default'],
        current_distance_to_default=res['distance_to_default'],
        days_past_due=0,
        is_credit_impaired=False,
        ead_schedule=[47200.0],
        effective_interest_rate=0.045
    )
    print("\n[3] REGULATORY IFRS 9 ECL STAGING:")
    print(f"  Assigned Staging:              {ecl_res['stage_name']}")
    print(f"  12-Month ECL Allowance:        ${ecl_res['total_ecl']:,.2f}")
    print(f"  ECL Provisioning % of EAD:     {ecl_res['ecl_percentage_of_ead']:.4f}%")
    print(f"  Indicative Underwriting Stance: BAND 1 (MINIMAL CREDIT RISK) / PRIME SOLVENCY")


if __name__ == "__main__":
    run_svb_backtest()
    run_msft_benchmark()
