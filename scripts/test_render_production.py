import sys
import time
import json
import httpx

BASE_URL = "https://aegis-engine-s96v.onrender.com"

def check_endpoint(client, method, path, **kwargs):
    url = f"{BASE_URL}{path}"
    print(f"\n---> {method} {url}")
    t0 = time.time()
    try:
        if method == "GET":
            res = client.get(url, timeout=60.0, **kwargs)
        else:
            res = client.post(url, timeout=60.0, **kwargs)
        dt = time.time() - t0
        print(f"Status: {res.status_code} ({dt:.2f}s)")
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Response: {res.text[:500]}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    with httpx.Client() as client:
        # 1. Health
        health = check_endpoint(client, "GET", "/health")
        print(f"Health Response: {health}")

        # 2. Companies Directory
        companies = check_endpoint(client, "GET", "/api/v1/market/companies")
        if companies:
            print(f"Directory: total {companies.get('total_count')} companies, {companies.get('automatic_count')} automatic filers.")
            sample = companies.get('companies', [])[:5]
            print(f"Sample companies: {sample}")

        # 3. Test AAPL, MSFT, NVDA
        for ticker in ["AAPL", "MSFT", "NVDA"]:
            data = check_endpoint(client, "GET", "/api/v1/market/borrower", params={"ticker": ticker})
            if data:
                print(f"=== Successfully retrieved {ticker} ({data.get('company_name')}) ===")
                print(f"  CIK: {data.get('cik')}, SIC: {data.get('sic_description')}")
                print(f"  Equity Market Cap: ${data.get('equity_value', {}).get('value'):,.1f}M (Source: {data.get('equity_value', {}).get('source')})")
                print(f"  Equity Volatility: {data.get('equity_volatility', {}).get('value', 0)*100:.1f}%")
                print(f"  Short-Term Debt: ${data.get('short_term_debt', {}).get('value'):,.1f}M (Source: {data.get('short_term_debt', {}).get('source')})")
                print(f"  Long-Term Debt: ${data.get('long_term_debt', {}).get('value'):,.1f}M")
                print(f"  Risk-Free Rate: {data.get('risk_free_rate', {}).get('value', 0)*100:.2f}% (Source: {data.get('risk_free_rate', {}).get('source')})")
                print(f"  Warnings: {data.get('warnings')}")
                
                # Check forensic analysis
                f_analysis = data.get('forensic_analysis')
                if f_analysis:
                    beneish = f_analysis.get('beneish', {})
                    altman = f_analysis.get('altman', {})
                    ccc = f_analysis.get('ccc', {})
                    print(f"  Forensics:")
                    print(f"    Beneish M-Score: {beneish.get('m_score'):.2f} (Manipulator Flag: {beneish.get('is_manipulator')})")
                    print(f"    Altman Z''-Score: {altman.get('z_double_prime'):.2f} (Zone: {altman.get('zone')})")
                    print(f"    Cash Conversion Cycle: {ccc.get('ccc_t'):.1f} days (Delta: {ccc.get('delta_ccc'):+.1f} days)")
                    print(f"    Composite Forensic Tier: {f_analysis.get('composite_forensic_risk')}")

if __name__ == "__main__":
    main()
