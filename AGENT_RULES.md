# Aegis Engine Agent Rules

All development, maintenance, and agent operations within the Aegis Quantitative Credit & Forensics Engine repository must strictly adhere to the following architectural rules:

1. **Explicit Accounting Line Disclosures:**
   - Never use silent defaults for missing accounting lines. All inputs must either be explicitly provided and validated or fail validation explicitly.

2. **Unclamped Solver Outputs & Convergence Verification:**
   - Never clamp solver outputs (`max(v_a, E)`) or hardcode `solver_converged=True`. Solver convergence must be determined strictly by objective function tolerance and residual tolerances.

3. **De-Agencified Internal Risk Bands:**
   - Never output external agency ratings (`AAA`, `BB`, `CCC`) or binary `APPROVE`/`REJECT` commands; use Aegis Internal Risk Bands (`Band 1` to `Band 7`) and non-binding indicative underwriting stances.

4. **Domain-Specific HTTP Error Handling:**
   - Raise HTTP 422 only for `(ValueError, ArithmeticError)` domain errors, never generic `Exception`. Unexpected system faults must bubble up or produce HTTP 500 errors.
