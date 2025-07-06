import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

/-!
  Self-contained definitions of the four primitive parameters used throughout the
  Yang-Mills proof.  We eliminate the dependency on the external Recognition-
  Science repository by defining the constants explicitly **and proving the basic
  properties that downstream files rely on**.

  Required downstream facts (searched via grep):

  * φ_eq_root   : φ² = φ + 1  and 1 < φ
  * E_coh_pos   : 0 < E_coh
  * q73_eq_73   : (q73 : ℤ) = 73
  * lambda_rec_pos   : 0 < lambda_rec

  Nothing else from the RSJ derivations is referenced, so providing these
  locally suffices.
-/

namespace RS.Param

open Real

/- Golden ratio – definition. -/
noncomputable def φ : ℝ := (1 + sqrt 5) / 2

/- The defining quadratic equation φ² = φ + 1. -/
lemma φ_eq_root : φ * φ = φ + 1 := by
  unfold φ
  -- φ = (1 + √5)/2
  -- Need to show: ((1 + √5)/2)² = (1 + √5)/2 + 1
  have h : sqrt 5 ^ 2 = 5 := sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)
  field_simp
  rw [h]
  ring

/- φ is greater than 1 (needed for positivity in gap formula). -/
lemma φ_gt_one : (1 : ℝ) < φ := by
  unfold φ
  have h1 : (0 : ℝ) < sqrt 5 := sqrt_pos.2 (by norm_num)
  have h2 : (2 : ℝ) < 1 + sqrt 5 := by linarith
  rw [div_lt_iff (by norm_num : (0 : ℝ) < 2)]
  linarith

/- Positive version needed downstream. -/
lemma φ_pos : 0 < φ := by linarith [φ_gt_one]

/- Minimal coherence energy constant.  The proof relies only on the literal. -/
def E_coh : ℝ := 0.090

lemma E_coh_pos : 0 < E_coh := by
  unfold E_coh; norm_num

/- The integer 73. -/
def q73 : ℕ := 73

lemma q73_eq_73 : (q73 : ℤ) = 73 := by
  simp [q73]

/- Recognition length constant (any positive value suffices downstream). -/
noncomputable def lambda_rec : ℝ := sqrt 2

lemma lambda_rec_pos : 0 < lambda_rec := by
  unfold lambda_rec; exact sqrt_pos.2 (by norm_num)

/- Convenience lemma exposing explicit value (used by numerical files) -/
lemma φ_value : φ = (1 + sqrt 5) / 2 := by rfl

/- Explicit literal for E_coh. -/
lemma E_coh_value : E_coh = 0.090 := rfl

end RS.Param
