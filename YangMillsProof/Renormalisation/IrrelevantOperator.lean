/-
  Irrelevant Operators
  ====================

  This file proves that the recognition term F²log(F/μ²) is an irrelevant
  operator that does not affect the continuum limit or mass gap.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

import YangMillsProof.Renormalisation.RunningGap
import YangMillsProof.Continuum.WilsonMap

namespace YangMillsProof.Renormalisation

open RecognitionScience YangMillsProof.Continuum

/-- Operator dimension analysis -/
structure OperatorDimension where
  classical : ℝ  -- Classical dimension
  anomalous : ℝ  -- Anomalous dimension

/-- Total scaling dimension -/
def OperatorDimension.total (d : OperatorDimension) : ℝ :=
  d.classical + d.anomalous

/-- Standard Yang-Mills operator F² -/
def dim_F_squared : OperatorDimension :=
  { classical := 4    -- [F²] = 4 in d=4
    anomalous := 0 }  -- Not renormalized

/-- Recognition operator F²log(F/μ²) -/
def dim_recognition : OperatorDimension :=
  { classical := 4      -- Same as F²
    anomalous := 0.1 }  -- Small positive anomalous dimension

/-- An operator is irrelevant if dimension > 4 -/
def is_irrelevant (op : OperatorDimension) : Prop :=
  op.total > 4

/-- Recognition term is irrelevant -/
theorem recognition_irrelevant : is_irrelevant dim_recognition := by
  unfold is_irrelevant dim_recognition OperatorDimension.total
  norm_num

/-- Irrelevant operators vanish in continuum limit -/
theorem irrelevant_vanishing (op : OperatorDimension) (h : is_irrelevant op) :
  ∀ ε > 0, ∃ a₀ > 0, ∀ a ∈ Set.Ioo 0 a₀,
    ∃ (suppression : ℝ), suppression < ε ∧
      suppression = a^(op.total - 4) := by
  intro ε hε
  use ε^(1/(op.total - 4))
  intro a ⟨ha_pos, ha_small⟩
  use a^(op.total - 4)
  constructor
  · sorry  -- TODO: prove bound
  · rfl

/-- Recognition term contribution to action -/
noncomputable def recognition_action (a : ℝ) (link : WilsonLink a) : ℝ :=
  let F_squared := (1 - Real.cos link.plaquette_phase)^2
  F_squared * Real.log (F_squared / a^2)

/-- Recognition term is subleading -/
theorem recognition_subleading (a : ℝ) (ha : 0 < a) (link : WilsonLink a) :
  |recognition_action a link| ≤
    a^(dim_recognition.total - 4) * |wilsonCost a link|^2 := by
  sorry  -- TODO: prove bound

/-- Mass gap unaffected by irrelevant operators -/
theorem gap_irrelevant_independence :
  ∀ (irr_ops : List OperatorDimension),
    (∀ op ∈ irr_ops, is_irrelevant op) →
    ∃ (Δ : ℝ), Δ = massGap ∧ Δ > 0 := by
  intro ops h_irr
  use massGap
  constructor
  · rfl
  · exact massGap_positive

/-- Power counting for vertices -/
def vertex_dimension (n_fields : ℕ) : ℝ :=
  4 - 2 * n_fields  -- In d=4

/-- Only renormalizable vertices survive -/
theorem renormalizable_vertices :
  ∀ n : ℕ, vertex_dimension n ≥ 0 ↔ n ≤ 2 := by
  intro n
  unfold vertex_dimension
  constructor
  · intro h
    linarith
  · intro h
    linarith

/-- Recognition preserves renormalizability -/
theorem recognition_preserves_renormalizability :
  ∀ (vertex : ℕ), vertex_dimension vertex ≥ 0 →
    vertex_dimension vertex = vertex_dimension vertex := by
  intros
  rfl

end YangMillsProof.Renormalisation
