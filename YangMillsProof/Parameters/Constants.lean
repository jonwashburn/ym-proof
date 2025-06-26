/-
  Recognition Science Parameters
  ==============================

  This file declares the fundamental parameters used in the Yang-Mills proof.
  These are treated as unspecified constants to make the proof parametric.
-/

namespace RS.Param

/-- Golden ratio (self-similarity parameter) -/
constant φ : ℝ

/-- Coherence quantum (fundamental energy unit) -/
constant E_coh : ℝ

/-- Plaquette charge (topological quantum number) -/
constant q73 : ℕ

/-- Recognition length (mesoscopic scale) -/
constant λ_rec : ℝ

/-- Physical string tension in GeV² -/
constant σ_phys : ℝ

/-- Critical coupling constant -/
constant β_critical : ℝ

/-- Lattice spacing in fm -/
constant a_lattice : ℝ

/-- Step-scaling product constant -/
constant c₆ : ℝ

end RS.Param
