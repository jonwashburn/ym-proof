# Yang-Mills Proof – Comprehensive Improvement Plan

_This file tracks every outstanding task (mathematical, physical, coding, and infrastructure) that must be completed before the repository can be considered a polished, publication-ready formal proof._  Each item has a priority tag:

• **[H] High** – blocks correctness claims or might hide an unsound step.  
• **[M] Medium** – desirable for rigour/readability but not unsound.  
• **[L] Low** – stylistic or infrastructure niceties.

---
## 1  Mathematical Rigour

| ID | Priority | File | Description |
|-----|----------|------|-------------|
| M-1 | H | `RG/ExactSolution.lean` | Replace heuristic lower-bound in `g_exact_approx` with a formal inequality (0.8 < g). Introduce analytic lemma or numeric bound. |
| M-2 | H | `RG/ExactSolution.lean` | Make `c_exact_bounds` prove the advertised interval (1.14, 1.20) instead of relying on positivity.  Use monotonicity + concrete numeric constants or interval arithmetic. |
| M-3 | H | `RG/ExactSolution.lean` | Re-prove `c_product_value` with tight bounds (≈7.55) rather than the loose product of upper bounds. |
| M-4 | M | `RG/ExactSolution.lean` | Break huge `ring_nf` steps in `g_exact_satisfies_rg` into helper lemmas for maintainability. |
| M-5 | M | `RG/StepScaling.lean` | Define `beta_one_loop` and restate `rg_flow_equation` against that function.  Remove the artificial equality `b₁*g⁵ = 0`. |
| M-6 | H | New | Collect all numerical constants/inequalities (π bounds, log estimates, etc.) in `Numerical/Lemmas.lean` so proofs call a single source of truth. |
| M-7 | L | project-wide | Audit doc-strings so theorem statements and comments match the weakened results (e.g. "tight bound <0.1" → now equality to 0). |

---
## 2  Physics Modelling & Place-holders

| ID | Priority | File | Description |
|-----|----------|------|-------------|
| P-1 | H | `Wilson/LedgerBridge.lean` | Implement realistic `plaquetteHolonomy` for SU(3) lattice gauge field.  Requires: link variable type, group multiplication along plaquette path, coercion to `SU(3)`. |
| P-2 | H | `Wilson/LedgerBridge.lean` | Define non-trivial `centreProject` and `centreCharge`.  Remove hard-coded `1`.
| P-3 | H | `Wilson/LedgerBridge.lean` | Restore meaningful version of `tight_bound_at_critical`; prove numeric bound once P-1/P-2 done. |
| P-4 | M | `Wilson/LedgerBridge.lean` | Calibrate `β_critical_derived` to lattice data; replace current crude inequality (10 < β < 12) with accurate proof connecting to Wilson β≈6.
| P-5 | M | `RG/ExactSolution.lean` | Verify constants `μ_ref`, `c_i`, and `c_product` numerically with exact evaluation once P-1/P-2 provide genuine `g_exact` region.

---
## 3  Numerical Verification & Automation

| ID | Priority | Description |
|-----|----------|-------------|
| N-1 | M | Add `test/` folder with Lean `#eval` or `lake exe` scripts comparing computed numbers (`c_product`, `Δ_phys_exact`, etc.) against expected intervals. |
| N-2 | M | Introduce CI (GitHub Actions) running `lake build` + numeric tests to guarantee repository stays sorry-free and within numeric tolerances. |
| N-3 | L | Implement interval-arithmetic tactic (`interval_arith`) or use `positivity`+`linarith` wrappers to shorten numeric inequality proofs.

---
## 4  Code Quality & Repository Hygiene

| ID | Priority | Description |
|-----|----------|-------------|
| C-1 | L | Split giant commits into thematic branches (Math, Physics, Infrastructure). |
| C-2 | L | Add `CONTRIBUTING.md` explaining coding conventions, placeholder policy, and how to run tests. |
| C-3 | L | Add automatic `lake env` pre-commit hook to reject new `sorry`. |

---
## 5  Documentation & Exposition

| ID | Priority | Description |
|-----|----------|-------------|
| D-1 | M | Expand repository README: build instructions, proof outline, status badge (CI), explanation of RSJ integration. |
| D-2 | L | Inline diagrams or comments explaining the one-loop solution and step-scaling math. |

---
### Completion Criteria 🔒
The proof will be considered _solid and complete_ when **all High-priority (H) items are closed**, all placeholder implementations are replaced by genuine mathematics/physics, and CI shows 0 sorries plus numeric tests passing.

Progress tracking:
```
[ ] M-1  [ ] M-2  [ ] M-3  [ ] M-4  [ ] M-5  [ ] M-6  [ ] M-7
[ ] P-1  [ ] P-2  [ ] P-3  [ ] P-4  [ ] P-5
[ ] N-1  [ ] N-2  [ ] N-3
[ ] C-1  [ ] C-2  [ ] C-3
[ ] D-1  [ ] D-2
```
Pull requests should reference the IDs (e.g. `Closes M-1, M-2`). 

lemma deriv_g_exact (hμ : μ ≠ 0)
  : deriv (g_exact μ₀ g₀) μ
    = (- b₀ * (g_exact μ₀ g₀ μ)^3) / μ