import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from services.market_data import load_borrower
from core.merton import MertonSolver
from core.forensics import calculate_forensics
from schemas.credit_models import MertonInput, ForensicInput

async def test_downstream():
    for sym in ['AAPL', 'MSFT', 'NVDA']:
        data = await load_borrower(sym)
        E = data['equity_value']['value']
        sigma_e = data['equity_volatility']['value']
        std = data['short_term_debt']['value']
        ltd = data['long_term_debt']['value']
        r = data['risk_free_rate']['value']
        
        m_in = MertonInput(
            equity_value=E,
            equity_volatility=sigma_e,
            short_term_debt=std,
            long_term_debt=ltd,
            risk_free_rate=r,
            maturity=1.0,
            loss_given_default=0.45
        )
        m_res = MertonSolver.solve(m_in)
        print(f"=== {sym} ({data['company_name']}) Downstream Results ===")
        print(f"  Asset Value (V_A): ${m_res.asset_value:,.1f}M")
        print(f"  Default Barrier (D): ${m_res.default_barrier:,.1f}M")
        print(f"  Distance to Default: {m_res.distance_to_default_physical:.2f} sigma")
        print(f"  Physical Default Prob: {m_res.default_probability_physical:.6e}")
        print(f"  Credit Spread: {m_res.credit_spread_bps:.2f} bps")
        
        if data.get('forensic_input'):
            f_res = calculate_forensics(ForensicInput(**data['forensic_input']))
            print(f"  Beneish M-Score: {f_res.beneish.m_score:.2f} (Manipulator Flag: {f_res.beneish.is_manipulator})")
            print(f"  Altman Z''-Score: {f_res.altman.z_double_prime:.2f} ({f_res.altman.zone})")
            print(f"  Cash Conversion Cycle: {f_res.ccc.ccc_t:.1f} days (Delta: {f_res.ccc.delta_ccc:+.1f} days)")
            print(f"  Composite Forensic Risk: {f_res.composite_forensic_risk}")
        print()

if __name__ == "__main__":
    asyncio.run(test_downstream())
