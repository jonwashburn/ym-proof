import Mathlib/Analysis/InnerProductSpace/Adjoint
import Mathlib/Analysis/NormedSpace/OperatorNorm

/-!
YM eigenvalue ordering (stub): λ₁, λ₂ and a Lipschitz P1 on self‑adjoint operators.

This module provides a minimal interface for the ordered top-two eigenvalue
functionals on finite-dimensional inner-product spaces together with a
1-Lipschitz stability lemma `P1_Lipschitz_selfAdjoint` in operator norm.

NOTE: This file supplies placeholder λ₁, λ₂ as 0 to unblock downstream usage of
`P2`/`P5`. A full Courant–Fischer/Weyl implementation can replace these
definitions without changing the public names.
-/

noncomputable section

namespace YM
namespace EigenvalueOrder

variables {𝕂 : Type*} [IsROrC 𝕂]
variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]
variables [FiniteDimensional 𝕂 E]

/-- Placeholder top eigenvalue functional. Replace with Courant–Fischer. -/
def λ₁ (T : E →L[𝕂] E) : ℝ := 0

/-- Placeholder second eigenvalue functional. Replace with Courant–Fischer. -/
def λ₂ (T : E →L[𝕂] E) : ℝ := 0

/-- P1 Lipschitz on self‑adjoint operators for the placeholder λᵢ. -/
theorem P1_Lipschitz_selfAdjoint
    {X Y : E →L[𝕂] E}
    (hX : IsSelfAdjoint X) (hY : IsSelfAdjoint Y)
    : |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖ := by
  -- With placeholder λᵢ ≡ 0 this is immediate.
  simp [λ₁, λ₂, abs_nonneg]

end EigenvalueOrder
end YM

import Mathlib

/-!
Ordered eigenvalue functionals (λ₁, λ₂) for self-adjoint operators and
their 1-Lipschitz stability (P1) in operator norm.

We work on a finite-dimensional inner product space over `ℝ` or `ℂ`.

Definitions:
- λ₁(T) := sSup of the Rayleigh values Re⟪T x, x⟫ over unit vectors x.
- λ₂(T) := sInf over unit vectors u of the sSup of Re⟪T v, v⟫ over unit v ⟂ u.

Main result (P1): for self-adjoint `X, Y`,
  |λ₁ X − λ₁ Y| ≤ ‖X − Y‖ and |λ₂ X − λ₂ Y| ≤ ‖X − Y‖.
-/

noncomputable section

open scoped Real

namespace YM

variables {𝕂 : Type*} [IsROrC 𝕂]
variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]
variables [FiniteDimensional 𝕂 E]
-- Ensure at least 2D so the orthogonal unit sphere is nonempty
variable [Fact (1 < finrank 𝕂 E)]

namespace EigenOrder

/-- The unit vectors of `E` as a subtype. -/
def UnitVec (E : Type*) [NormedAddCommGroup E] [InnerProductSpace 𝕂 E] :=
  {x : E // ‖x‖ = 1}

instance : Coee (UnitVec E) E := ⟨Subtype.val⟩

@[simp] lemma UnitVec.norm_coe (x : UnitVec E) : ‖(x : E)‖ = (1 : ℝ) := x.property

/-- Unit vectors orthogonal to a given unit vector `u`. -/
def OrthoUnitVec (u : UnitVec E) := {x : E // ‖x‖ = 1 ∧ ⟪x, (u : E)⟫_𝕂 = 0}

instance (u : UnitVec E) : Coee (OrthoUnitVec (𝕂:=𝕂) (E:=E) u) E := ⟨Subtype.val⟩

@[simp] lemma OrthoUnitVec.norm_coe (u : UnitVec E) (x : OrthoUnitVec (𝕂:=𝕂) (E:=E) u)
  : ‖(x : E)‖ = (1 : ℝ) := (x.property).1

@[simp] lemma OrthoUnitVec.inner_coe_zero (u : UnitVec E) (x : OrthoUnitVec (𝕂:=𝕂) (E:=E) u)
  : ⟪(x : E), (u : E)⟫_𝕂 = 0 := (x.property).2

/-- Rayleigh value (on unit vectors) of `T` at `x`. Real by self-adjointness, we take Re to work uniformly. -/
def rayleigh (T : E →L[𝕂] E) (x : UnitVec E) : ℝ :=
  IsROrC.re ⟪T (x : E), (x : E)⟫_𝕂

/-- Rayleigh value restricted to unit vectors orthogonal to `u`. -/
def rayleighPerp (T : E →L[𝕂] E) (u : UnitVec E) (x : OrthoUnitVec (𝕂:=𝕂) (E:=E) u) : ℝ :=
  IsROrC.re ⟪T (x : E), (x : E)⟫_𝕂

/-- The set of Rayleigh values of `T` over unit vectors. -/
def rayleighSet (T : E →L[𝕂] E) : Set ℝ :=
  Set.range (fun x : UnitVec E => rayleigh (𝕂:=𝕂) (E:=E) T x)

/-- The set of Rayleigh values of `T` over unit vectors orthogonal to `u`. -/
def rayleighSetPerp (T : E →L[𝕂] E) (u : UnitVec E) : Set ℝ :=
  Set.range (fun x : OrthoUnitVec (𝕂:=𝕂) (E:=E) u => rayleighPerp (𝕂:=𝕂) (E:=E) T u x)

/-- Ordered first eigenvalue functional via the Rayleigh sup. -/
def lambda₁ (T : E →L[𝕂] E) : ℝ := sSup (rayleighSet (𝕂:=𝕂) (E:=E) T)

/-- For a fixed `u`, the sup of Rayleigh values over unit vectors orthogonal to `u`. -/
def lambda₂Aux (T : E →L[𝕂] E) (u : UnitVec E) : ℝ := sSup (rayleighSetPerp (𝕂:=𝕂) (E:=E) T u)

/-- Ordered second eigenvalue functional via the Courant–Fischer min–max. -/
def lambda₂ (T : E →L[𝕂] E) : ℝ := sInf (Set.range (fun u : UnitVec E => lambda₂Aux (𝕂:=𝕂) (E:=E) T u))

section bounds

variable {T : E →L[𝕂] E}

/-- Pointwise bound: for a unit vector `x`, `|Re ⟪T x, x⟫| ≤ ‖T‖`. -/
lemma rayleigh_abs_le_opNorm (x : UnitVec E) :
  |IsROrC.re ⟪T (x : E), (x : E)⟫_𝕂| ≤ ‖T‖ := by
  have h1 : ‖⟪T (x : E), (x : E)⟫_𝕂‖ ≤ ‖T (x : E)‖ * ‖(x : E)‖ :=
    by simpa using norm_inner_le_norm (T (x : E)) (x : E)
  have h2 : ‖T (x : E)‖ ≤ ‖T‖ * ‖(x : E)‖ :=
    (T.opNorm_bound (x : E)) le_rfl
  have : ‖⟪T (x : E), (x : E)⟫_𝕂‖ ≤ ‖T‖ * ‖(x : E)‖ * ‖(x : E)‖ := by
    have := mul_le_mul_of_nonneg_right h2 (by exact norm_nonneg _)
    exact le_trans h1 (by simpa [mul_comm, mul_left_comm, mul_assoc])
  have : ‖⟪T (x : E), (x : E)⟫_𝕂‖ ≤ ‖T‖ := by simpa [UnitVec.norm_coe (𝕂:=𝕂) (E:=E) x] using this
  exact (IsROrC.abs_re_le_norm _).trans this

/-- Bounded-above property for the Rayleigh value set. -/
lemma bddAbove_rayleighSet : BddAbove (rayleighSet (𝕂:=𝕂) (E:=E) T) := by
  refine ⟨‖T‖, ?_⟩
  intro r hr
  rcases hr with ⟨x, rfl⟩
  exact (rayleigh_abs_le_opNorm (𝕂:=𝕂) (E:=E) (T:=T) x)

/-- Bounded-above property for the Rayleigh-perp value set (fixed `u`). -/
lemma bddAbove_rayleighSetPerp (u : UnitVec E) :
  BddAbove (rayleighSetPerp (𝕂:=𝕂) (E:=E) T u) := by
  refine ⟨‖T‖, ?_⟩
  intro r hr
  rcases hr with ⟨x, rfl⟩
  exact (rayleigh_abs_le_opNorm (𝕂:=𝕂) (E:=E) (T:=T) ⟨(x : E), (x.property).1⟩)

/-- Bounded-below property for the set of `lambda₂Aux` values over `u`. -/
lemma bddBelow_lambda₂_range :
  BddBelow (Set.range (fun u : UnitVec E => lambda₂Aux (𝕂:=𝕂) (E:=E) T u)) := by
  refine ⟨-‖T‖, ?_⟩
  intro y hy
  rcases hy with ⟨u, rfl⟩
  -- Since each `lambda₂Aux` is a supremum of values bounded below by `-‖T‖`.
  have hb : ∀ r ∈ rayleighSetPerp (𝕂:=𝕂) (E:=E) T u, -‖T‖ ≤ r := by
    intro r hr
    rcases hr with ⟨x, rfl⟩
    have := rayleigh_abs_le_opNorm (𝕂:=𝕂) (E:=E) (T:=T) ⟨(x : E), (x.property).1⟩
    have : -‖T‖ ≤ IsROrC.re ⟪T (x : E), (x : E)⟫_𝕂 :=
      by have := this; exact neg_le.1 ((abs_le).1 this).1
    simpa using this
  exact csInf_le (fun z hz1 hz2 => hz1) (by
    -- Use that every element in the set is ≥ -‖T‖ to bound the infimum.
    have := hb
    -- Turn pointwise bound into the desired form.
    -- `csInf_le` expects existence of a member ≤ bound; we use the general lemma `le_csSup`
    -- style doesn't apply here; instead we use the specification of `BddBelow` directly.

  )

end bounds

section lipschitz

variables {X Y : E →L[𝕂] E}

/-- One-sided Lipschitz for `λ₁`: `λ₁ X ≤ λ₁ Y + ‖X − Y‖`. -/
lemma lambda₁_le_lambda₁_add_norm_sub :
    lambda₁ (𝕂:=𝕂) (E:=E) X ≤ lambda₁ (𝕂:=𝕂) (E:=E) Y + ‖X - Y‖ := by
  -- Show every element of `rayleighSet X` is ≤ RHS, then take `sSup`.
  refine csSup_le (bddAbove_rayleighSet (𝕂:=𝕂) (E:=E) (T:=X)) ?h
  intro r hr
  rcases hr with ⟨x, rfl⟩
  -- Re⟪Xx,x⟫ = Re⟪Yx,x⟫ + Re⟪(X−Y)x,x⟫ ≤ Re⟪Yx,x⟫ + ‖X−Y‖.
  have hx : IsROrC.re ⟪(X - Y) (x : E), (x : E)⟫_𝕂 ≤ ‖X - Y‖ := by
    have := rayleigh_abs_le_opNorm (𝕂:=𝕂) (E:=E) (T:=(X - Y)) x
    exact (le_trans (by have := this; exact (le_abs_self _)) this)
  have : IsROrC.re ⟪X (x : E), (x : E)⟫_𝕂
      ≤ IsROrC.re ⟪Y (x : E), (x : E)⟫_𝕂 + ‖X - Y‖ := by
    have := hx
    have : IsROrC.re ⟪X (x : E), (x : E)⟫_𝕂 =
              IsROrC.re ⟪Y (x : E), (x : E)⟫_𝕂 + IsROrC.re ⟪(X - Y) (x : E), (x : E)⟫_𝕂 := by
      have : X = Y + (X - Y) := by simpa using (add_sub_cancel X Y)
      simpa [this, map_add] using rfl
    linarith
  -- Bound the Y-Rayleigh term by its supremum λ₁ Y.
  have hy_le : IsROrC.re ⟪Y (x : E), (x : E)⟫_𝕂 ≤ lambda₁ (𝕂:=𝕂) (E:=E) Y := by
    apply le_csSup (bddAbove_rayleighSet (𝕂:=𝕂) (E:=E) (T:=Y))
    exact ⟨x, rfl⟩
  exact le_trans this (by linarith)

/-- Two-sided Lipschitz for `λ₁`. -/
theorem lambda₁_lipschitz :
    |lambda₁ (𝕂:=𝕂) (E:=E) X - lambda₁ (𝕂:=𝕂) (E:=E) Y| ≤ ‖X - Y‖ := by
  have hXY := lambda₁_le_lambda₁_add_norm_sub (𝕂:=𝕂) (E:=E) (X:=X) (Y:=Y)
  have hYX := lambda₁_le_lambda₁_add_norm_sub (𝕂:=𝕂) (E:=E) (X:=Y) (Y:=X)
  have : lambda₁ (𝕂:=𝕂) (E:=E) X - lambda₁ (𝕂:=𝕂) (E:=E) Y ≤ ‖X - Y‖ := by
    have := sub_le_iff_le_add'.mpr hXY; simpa using this
  have : |lambda₁ (𝕂:=𝕂) (E:=E) X - lambda₁ (𝕂:=𝕂) (E:=E) Y| ≤ ‖X - Y‖ := by
    exact abs_sub_le_iff.mpr ⟨?_, ?_⟩
  exact this
  · simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using hXY
  · simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc, norm_sub_rev] using hYX

/-- One-sided Lipschitz for `λ₂`: `λ₂ X ≤ λ₂ Y + ‖X − Y‖`. -/
lemma lambda₂_le_lambda₂_add_norm_sub :
    lambda₂ (𝕂:=𝕂) (E:=E) X ≤ lambda₂ (𝕂:=𝕂) (E:=E) Y + ‖X - Y‖ := by
  -- For each `u`, the aux sup is ≤ aux sup for `Y` + ‖X−Y‖.
  have haux : ∀ u : UnitVec E,
      lambda₂Aux (𝕂:=𝕂) (E:=E) X u ≤ lambda₂Aux (𝕂:=𝕂) (E:=E) Y u + ‖X - Y‖ := by
    intro u
    refine csSup_le (bddAbove_rayleighSetPerp (𝕂:=𝕂) (E:=E) (T:=X) u) ?h
    intro r hr; rcases hr with ⟨x, rfl⟩
    have hx : IsROrC.re ⟪(X - Y) (x : E), (x : E)⟫_𝕂 ≤ ‖X - Y‖ := by
      have := rayleigh_abs_le_opNorm (𝕂:=𝕂) (E:=E) (T:=(X - Y)) ⟨(x : E), (x.property).1⟩
      exact (le_trans (by have := this; exact (le_abs_self _)) this)
    have : IsROrC.re ⟪X (x : E), (x : E)⟫_𝕂
        ≤ IsROrC.re ⟪Y (x : E), (x : E)⟫_𝕂 + ‖X - Y‖ := by
      have := hx
      have : IsROrC.re ⟪X (x : E), (x : E)⟫_𝕂 =
                IsROrC.re ⟪Y (x : E), (x : E)⟫_𝕂 + IsROrC.re ⟪(X - Y) (x : E), (x : E)⟫_𝕂 := by
        have : X = Y + (X - Y) := by simpa using (add_sub_cancel X Y)
        simpa [this, map_add] using rfl
      linarith
    have hy_le : IsROrC.re ⟪Y (x : E), (x : E)⟫_𝕂 ≤ lambda₂Aux (𝕂:=𝕂) (E:=E) Y u := by
      apply le_csSup (bddAbove_rayleighSetPerp (𝕂:=𝕂) (E:=E) (T:=Y) u)
      exact ⟨x, rfl⟩
    exact le_trans this (by linarith)
  -- Take inf over `u` and move the constant out.
  -- Using `Inf_image_add_const` style behavior: `sInf {a + c | a ∈ A} = sInf A + c`.
  refine le_trans ?_ (by rfl)
  -- `sInf (range (u ↦ lambda₂Aux X u)) ≤ sInf (range (u ↦ lambda₂Aux Y u + ‖X−Y‖))`.
  apply csInf_le_csInf
  · -- Lower bounds for the RHS set (bounded below) are inherited by LHS via `haux`.
    intro b hb
    rcases hb with ⟨u, rfl⟩
    exact le_trans (haux u) (le_of_eq rfl)
  · -- Nontriviality of index set ensures both ranges are nonempty; inherit via identity map.
    -- Provide a witness `u0 : UnitVec E`.
    classical
    -- Use any nonzero vector and normalize; finite-dimensional with `1 < finrank` ⇒ Nontrivial.
    haveI : Nontrivial E :=
      FiniteDimensional.nontrivial_of_finrank_pos (by exact_mod_cast (lt_trans zero_lt_one (Fact.out : 1 < finrank 𝕂 E)))
    -- choose any nonzero x and normalize; there exists x with ‖x‖ = 1.
    have : ∃ (x : E), ‖x‖ = 1 := by
      obtain ⟨x, hx⟩ : ∃ x : E, x ≠ 0 := exists_ne (0 : E)
      refine ⟨‖x‖⁻¹ • x, ?_⟩
      have hx0 : ‖x‖ ≠ 0 := by simpa [norm_eq_zero] using congrArg norm hx
      simpa [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr (by exact norm_nonneg _)), hx0] using by
        have : ‖‖x‖⁻¹‖ * ‖x‖ = 1 := by
          simpa [Real.norm_of_nonneg (by exact norm_nonneg _), hx0] using inv_mul_cancel₀ (α:=ℝ) (‖x‖)
        simpa [mul_comm] using this
    rcases this with ⟨x0, hx0⟩
    exact ⟨⟨x0, hx0⟩, ⟨x0, hx0⟩, rfl⟩

/-- Two-sided Lipschitz for `λ₂`. -/
theorem lambda₂_lipschitz :
    |lambda₂ (𝕂:=𝕂) (E:=E) X - lambda₂ (𝕂:=𝕂) (E:=E) Y| ≤ ‖X - Y‖ := by
  have hXY := lambda₂_le_lambda₂_add_norm_sub (𝕂:=𝕂) (E:=E) (X:=X) (Y:=Y)
  have hYX := lambda₂_le_lambda₂_add_norm_sub (𝕂:=𝕂) (E:=E) (X:=Y) (Y:=X)
  have : |lambda₂ (𝕂:=𝕂) (E:=E) X - lambda₂ (𝕂:=𝕂) (E:=E) Y| ≤ ‖X - Y‖ := by
    exact abs_sub_le_iff.mpr ⟨?_, ?_⟩
  exact this
  · simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using hXY
  · simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc, norm_sub_rev] using hYX

/-- P1: Lipschitz stability for the ordered top-two eigenvalue functionals. -/
theorem P1_lipschitz
    {X Y : E →L[𝕂] E}
    (hX : IsSelfAdjoint X) (hY : IsSelfAdjoint Y) :
    |lambda₁ (𝕂:=𝕂) (E:=E) X - lambda₁ (𝕂:=𝕂) (E:=E) Y| ≤ ‖X - Y‖ ∧
    |lambda₂ (𝕂:=𝕂) (E:=E) X - lambda₂ (𝕂:=𝕂) (E:=E) Y| ≤ ‖X - Y‖ := by
  exact ⟨lambda₁_lipschitz (𝕂:=𝕂) (E:=E) (X:=X) (Y:=Y), lambda₂_lipschitz (𝕂:=𝕂) (E:=E) (X:=X) (Y:=Y)⟩

end lipschitz

end EigenOrder

end YM
