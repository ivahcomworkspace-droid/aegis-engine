"""
Number formats shared by the API summary and the credit memo. They mirror the dashboard's
formatMoneyM and formatPD (frontend/index.html), so a figure reads the same on screen and on paper.
Money is in $ millions, the engine's unit throughout.
"""

import math


def format_money_m(value_m: float) -> str:
    """$ millions as $1.23T, $4.2B, $276.5M, $5.5k or $18."""
    v = float(value_m or 0.0)
    if not math.isfinite(v):
        return "n/a"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}T"
    if a >= 1e3:
        decimals = 0 if a >= 1e5 else (1 if a >= 1e4 else 2)
        return f"{sign}${a / 1e3:.{decimals}f}B"
    if a >= 1:
        return f"{sign}${a:.1f}M"
    if a >= 1e-3:
        return f"{sign}${a * 1e3:.1f}k"
    return f"{sign}${round(a * 1e6):,}"


def format_pd(pd: float) -> str:
    """Probability of default as a percentage, keeping tiny values visible: 2.315%, 0.0050%, 3.73e-5%."""
    p = float(pd or 0.0)
    if not math.isfinite(p) or p <= 0:
        return "0.000%"
    pct = p * 100.0
    if pct >= 0.01:
        return f"{pct:.3f}%"
    if pct >= 0.0001:
        return f"{pct:.4f}%"
    mantissa, exponent = f"{pct:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}%"


def format_pct(fraction: float, decimals: int = 1) -> str:
    """A decimal fraction as a percentage: 0.2532 -> 25.3%."""
    return f"{float(fraction) * 100.0:.{decimals}f}%"
