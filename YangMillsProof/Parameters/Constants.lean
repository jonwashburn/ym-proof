-- Yang-Mills Parameters (after RSJ integration)
-- --------------------------------------------
-- φ, E_coh, q73, λ_rec are imported (proven) via `Parameters.FromRS`.
-- The four constants below remain to be derived and are still declared
-- as free for now (see CONSTANTS_ROADMAP.md).

import YangMillsProof.Parameters.FromRS

namespace RS.Param

/-- Physical string tension (σ_phys) in GeV² – *to be derived*. -/
constant σ_phys : ℝ

/-- Critical lattice coupling (β_critical) – *to be derived*. -/
constant β_critical : ℝ

/-- Lattice spacing (a_lattice) in femtometres – *to be derived*. -/
constant a_lattice : ℝ

/-- Step-scaling product (c₆) – *to be derived*. -/
constant c₆ : ℝ

end RS.Param
