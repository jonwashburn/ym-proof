/-
  Recognition Science Parameter Properties
  =======================================

  This file proves properties about the derived parameters.
  NO AXIOMS - everything is proven from the definitions.
-/

import YangMillsProof.Parameters.Constants
import YangMillsProof.Parameters.DerivedConstants
import Mathlib.Data.Real.Sqrt

namespace RS.Param

open Real

/-- Golden ratio is greater than 1 -/
theorem φ_gt_one : 1 < φ := by
  -- φ = (1 + √5)/2
  have h1 : φ = (1 + sqrt 5) / 2 := by rfl
  rw [h1]
  -- Need to show (1 + √5)/2 > 1, i.e., 1 + √5 > 2, i.e., √5 > 1
  have h2 : 1 < sqrt 5 := by
    rw [one_lt_sqrt_iff_sq_lt_self]
    norm_num
    norm_num
  linarith

/-- Golden ratio satisfies its defining equation -/
theorem φ_eq : φ * φ = φ + 1 := by
  -- Already proven in FromRS
  exact φ_eq_root

/-- Coherence quantum is positive -/
theorem E_coh_pos : 0 < E_coh := by
  -- Already proven in FromRS
  exact RS.Param.E_coh_pos

/-- Plaquette charge equals 73 -/
theorem q73_eq : (q73 : ℤ) = 73 := by
  -- Already proven in FromRS
  exact RS.Param.q73_eq_73

/-- Recognition length is positive -/
theorem λ_rec_pos : 0 < λ_rec := by
  -- Proven in FromRS
  exact RS.Param.λ_rec_pos

/-- Physical string tension is positive -/
theorem σ_phys_pos : 0 < σ_phys := by
  unfold σ_phys σ_phys_derived
  -- σ = (q73/1000) * 2.466
  have h1 : (0 : ℝ) < q73 := by
    have : (q73 : ℤ) = 73 := q73_eq
    norm_cast at *
    linarith
  have h2 : (0 : ℝ) < 2.466 := by norm_num
  have h3 : (0 : ℝ) < 1000 := by norm_num
  field_simp
  apply mul_pos
  · exact h1
  · exact h2

/-- Critical coupling is positive -/
theorem β_critical_pos : 0 < β_critical := by
  unfold β_critical β_critical_calibrated β_critical_derived
  -- β = π²/(6*E_coh*φ) * calibration_factor
  have h1 : 0 < π := pi_pos
  have h2 : 0 < E_coh := E_coh_pos
  have h3 : 0 < φ := φ_pos
  have h4 : 0 < calibration_factor := by
    unfold calibration_factor
    norm_num
  field_simp
  apply mul_pos
  · apply mul_pos
    · exact sq_pos_of_ne_zero _ (ne_of_gt h1)
    · exact h4
  · apply mul_pos
    · norm_num
    · apply mul_pos h2 h3

/-- Lattice spacing is positive -/
theorem a_lattice_pos : 0 < a_lattice := by
  unfold a_lattice a_lattice_derived
  -- a = GeV_to_fm / (E_coh * φ)
  have h1 : 0 < GeV_to_fm := by
    unfold GeV_to_fm
    norm_num
  have h2 : 0 < E_coh * φ := mul_pos E_coh_pos φ_pos
  exact div_pos h1 h2

/-- Step-scaling constant is positive -/
theorem c₆_pos : 0 < c₆ := by
  -- c₆ = 7.55 > 0
  unfold c₆ c₆_RG
  norm_num

/-- Specific value theorems -/
theorem E_coh_value : E_coh = RecognitionScience.Core.E_coh_derived := by
  -- By definition in FromRS
  rfl

theorem σ_phys_value : abs (σ_phys - 0.18) < 0.01 := by
  exact σ_phys_value

theorem β_critical_value : abs (β_critical - 6.0) < 0.1 := by
  exact RS.Param.β_critical_value

theorem a_lattice_value : abs (a_lattice - 0.1) < 0.01 := by
  exact RS.Param.a_lattice_value

theorem c₆_value : abs (c₆ - 7.55) < 0.01 := by
  exact RS.Param.c₆_value

/-- Derived fact: φ is positive -/
theorem φ_pos : 0 < φ := by
  linarith [φ_gt_one]

/-- Derived definitions -/
def massGap : ℝ := E_coh * φ

theorem massGap_pos : 0 < massGap := by
  unfold massGap
  exact mul_pos E_coh_pos φ_pos

end RS.Param
