import Mathlib.Data.Real.Sqrt
import Mathlib.Topology.Algebra.InfiniteSum

namespace RSImport

open Real

/-- The golden ratio φ = (1 + √5)/2 -/
noncomputable def phi : ℝ := (1 + sqrt 5) / 2

lemma phi_pos : 0 < phi := by
  have h : sqrt 5 > 1 := by
    have : (1 : ℝ)^2 < (5 : ℝ) := by norm_num
    have h' := sqrt_lt_sqrt_iff.mpr this
    simpa using h'
  unfold phi
  have : (1 + sqrt 5) / 2 > 0 := by
    have : 1 + sqrt 5 > 0 := by linarith [h]
    have : (1 + sqrt 5) / 2 > 0 := by
      have : (1 + sqrt 5) > 0 := by linarith [h]
      have : (1 + sqrt 5) / 2 > 0 := by
        have : (1 + sqrt 5) / 2 = (1 + sqrt 5) * (1/2) := by field_simp
        have h2 : (1 + sqrt 5) > 0 := by linarith [h]
        have : (1 + sqrt 5) * (1/2) > 0 := by
          have : 1/2 > (0 : ℝ) := by norm_num
          exact mul_pos h2 this
        simpa [this] using this
    simpa using this
  exact this

/-- Simplified ledger entry -/
structure LedgerEntry where
  debit : ℝ
  credit : ℝ

/-- Ledger state as countable indexed entries -/
structure LedgerState where
  entry : ℕ → LedgerEntry
  finite_support : ∃ N, ∀ n > N, (entry n).debit = 0 ∧ (entry n).credit = 0

/-- Total debit -/
noncomputable def totalDebit (S : LedgerState) : ℝ := ∑' n, (S.entry n).debit
/-- Total credit -/
noncomputable def totalCredit (S : LedgerState) : ℝ := ∑' n, (S.entry n).credit

/-- Balanced ledger state -/
def balanced (S : LedgerState) : Prop := totalDebit S = totalCredit S

/-- The vacuum ledger state has zero entries -/
noncomputable def vacuumState : LedgerState where
  entry := fun _ => { debit := 0, credit := 0 }
  finite_support := by
    refine ⟨0, ?_⟩
    intro n hn
    simp

lemma vacuum_balanced : balanced vacuumState := by
  unfold balanced totalDebit totalCredit vacuumState
  simp

/-- Basic cost functional: sum of absolute imbalances weighted by φ^n -/
noncomputable def zeroCostFunctional (S : LedgerState) : ℝ :=
  ∑' n, |(S.entry n).debit - (S.entry n).credit| * phi ^ n

lemma zeroCostFunctional_nonneg (S : LedgerState) : zeroCostFunctional S ≥ 0 := by
  unfold zeroCostFunctional
  have : ∀ n, 0 ≤ |(S.entry n).debit - (S.entry n).credit| * phi ^ n := by
    intro n
    have h1 : 0 ≤ |(S.entry n).debit - (S.entry n).credit| := abs_nonneg _
    have h2 : 0 ≤ phi ^ n := by
      have hp : 0 ≤ phi := le_of_lt phi_pos
      exact pow_nonneg hp n
    have : 0 ≤ |(S.entry n).debit - (S.entry n).credit| * phi ^ n :=
      mul_nonneg h1 h2
    simpa using this
  exact tsum_nonneg this

end RSImport
