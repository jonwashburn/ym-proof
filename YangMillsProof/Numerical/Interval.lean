/-
  Interval Arithmetic for Numerical Verification
  =============================================

  A lightweight interval arithmetic library for verifying numerical bounds.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Numerical.Constants

namespace YangMillsProof.Numerical

/-- A closed rational interval [lo, hi] -/
structure Interval where
  lo : ℚ
  hi : ℚ
  le : lo ≤ hi

/-- Membership in an interval -/
def Interval.mem (x : ℝ) (I : Interval) : Prop :=
  (I.lo : ℝ) ≤ x ∧ x ≤ (I.hi : ℝ)

notation x " ∈ᵢ " I => Interval.mem x I

/-- Construct interval from bounds with proof -/
def Interval.mk' (lo hi : ℚ) (h : lo ≤ hi := by norm_num) : Interval :=
  ⟨lo, hi, h⟩

/-- Soundness: interval membership gives bounds -/
theorem Interval.sound {x : ℝ} {I : Interval} (h : x ∈ᵢ I) :
  (I.lo : ℝ) ≤ x ∧ x ≤ (I.hi : ℝ) := h

/-- Addition of intervals -/
def Interval.add (I₁ I₂ : Interval) : Interval :=
  ⟨I₁.lo + I₂.lo, I₁.hi + I₂.hi, add_le_add I₁.le I₂.le⟩

lemma Interval.add_mem {x y : ℝ} {I₁ I₂ : Interval}
  (hx : x ∈ᵢ I₁) (hy : y ∈ᵢ I₂) :
  x + y ∈ᵢ I₁.add I₂ := by
  unfold mem add
  simp only
  constructor
  · exact add_le_add hx.1 hy.1
  · exact add_le_add hx.2 hy.2

/-- Subtraction of intervals -/
def Interval.sub (I₁ I₂ : Interval) : Interval :=
  ⟨I₁.lo - I₂.hi, I₁.hi - I₂.lo, sub_le_sub I₁.le I₂.le⟩

lemma Interval.sub_mem {x y : ℝ} {I₁ I₂ : Interval}
  (hx : x ∈ᵢ I₁) (hy : y ∈ᵢ I₂) :
  x - y ∈ᵢ I₁.sub I₂ := by
  unfold mem sub
  simp only
  constructor
  · calc (I₁.lo - I₂.hi : ℝ) = I₁.lo - I₂.hi := by norm_cast
      _ ≤ x - y := sub_le_sub hx.1 hy.2
  · calc x - y ≤ I₁.hi - I₂.lo := sub_le_sub hx.2 hy.1
      _ = (I₁.hi - I₂.lo : ℝ) := by norm_cast

/-- Multiplication of intervals (positive case) -/
def Interval.mul_pos (I₁ I₂ : Interval) (h₁ : 0 < I₁.lo) (h₂ : 0 < I₂.lo) : Interval :=
  ⟨I₁.lo * I₂.lo, I₁.hi * I₂.hi, mul_le_mul I₁.le I₂.le (le_of_lt h₂) (le_of_lt h₁)⟩

lemma Interval.mul_pos_mem {x y : ℝ} {I₁ I₂ : Interval}
  (hx : x ∈ᵢ I₁) (hy : y ∈ᵢ I₂) (h₁ : 0 < I₁.lo) (h₂ : 0 < I₂.lo) :
  x * y ∈ᵢ I₁.mul_pos I₂ h₁ h₂ := by
  have hx_pos : 0 < x := lt_of_lt_of_le (by exact_mod_cast h₁) hx.1
  have hy_pos : 0 < y := lt_of_lt_of_le (by exact_mod_cast h₂) hy.1
  unfold mem mul_pos
  simp only
  constructor
  · exact mul_le_mul hx.1 hy.1 (le_of_lt hy_pos) (le_of_lt (by exact_mod_cast h₁))
  · exact mul_le_mul hx.2 hy.2 (le_of_lt (by exact_mod_cast h₂)) (le_of_lt hx_pos)

/-- Division of intervals (positive denominator) -/
def Interval.div_pos (I₁ I₂ : Interval) (h : 0 < I₂.lo) : Interval :=
  ⟨I₁.lo / I₂.hi, I₁.hi / I₂.lo, div_le_div_of_le_left I₁.le (by exact_mod_cast h) I₂.le⟩

lemma Interval.div_pos_mem {x y : ℝ} {I₁ I₂ : Interval}
  (hx : x ∈ᵢ I₁) (hy : y ∈ᵢ I₂) (h : 0 < I₂.lo) :
  x / y ∈ᵢ I₁.div_pos I₂ h := by
  have hy_pos : 0 < y := lt_of_lt_of_le (by exact_mod_cast h) hy.1
  unfold mem div_pos
  simp only
  constructor
  · calc (I₁.lo / I₂.hi : ℝ) = I₁.lo / I₂.hi := by norm_cast
      _ ≤ x / y := div_le_div_of_le_left hx.1 hy_pos hy.2
  · calc x / y ≤ I₁.hi / I₂.lo := div_le_div hx.2 (le_of_lt hy_pos) (by exact_mod_cast h) hy.1
      _ = (I₁.hi / I₂.lo : ℝ) := by norm_cast

/-- Interval for a single rational -/
def Interval.singleton (q : ℚ) : Interval :=
  ⟨q, q, le_refl q⟩

lemma Interval.singleton_mem (q : ℚ) : (q : ℝ) ∈ᵢ singleton q := by
  unfold mem singleton
  simp only [le_refl, and_self]

/-- Known interval bounds for constants -/
@[simp] lemma pi_interval : Real.pi ∈ᵢ Interval.mk' (314/100) (316/100) := by
  constructor
  · have : (314/100 : ℝ) < Real.pi := by norm_num; exact Real.pi_gt_314_div_100
    linarith
  · have : Real.pi < (22/7 : ℝ) := Real.pi_lt_22_div_7
    have : (22/7 : ℝ) < (316/100 : ℝ) := by norm_num
    linarith

@[simp] lemma log2_interval : Real.log 2 ∈ᵢ Interval.mk' (6931/10000) (6932/10000) := by
  -- We use that log 2 ≈ 0.693147...
  -- Import bounds from Constants
  have h := YangMillsProof.Numerical.Constants.log_two_bounds
  constructor
  · -- 0.6931 < log 2
    calc (6931/10000 : ℝ) = 0.6931 := by norm_num
      _ < Real.log 2 := h.1
  · -- log 2 < 0.6932
    calc Real.log 2 < 0.6932 := h.2
      _ = (6932/10000 : ℝ) := by norm_num

@[simp] lemma sqrt5_interval : Real.sqrt 5 ∈ᵢ Interval.mk' (2236/1000) (2237/1000) := by
  -- We use that √5 ≈ 2.236067...
  -- Lower bound: 2.236 < √5
  -- Upper bound: √5 < 2.237
  constructor
  · -- 2.236 < √5
    have h : (2236/1000)^2 < 5 := by norm_num
    have := Real.sq_lt_sq (by norm_num : (0 : ℝ) ≤ 2236/1000) (Real.sqrt_nonneg 5)
    rw [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)] at this
    exact this.mp h
  · -- √5 < 2.237
    have h : 5 < (2237/1000)^2 := by norm_num
    have := Real.sq_lt_sq (Real.sqrt_nonneg 5) (by norm_num : (0 : ℝ) ≤ 2237/1000)
    rw [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)] at this
    exact this.mpr h

/-- Interval arithmetic tactic -/
syntax "interval_arith" : tactic

macro_rules
  | `(tactic| interval_arith) => `(tactic|
      repeat (first
        | apply Interval.add_mem
        | apply Interval.sub_mem
        | apply Interval.mul_pos_mem <;> norm_num
        | apply Interval.div_pos_mem <;> norm_num
        | apply Interval.singleton_mem
        | exact pi_interval
        | exact log2_interval
        | exact sqrt5_interval
        | assumption))

end YangMillsProof.Numerical
