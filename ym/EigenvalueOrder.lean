import Mathlib
import Mathlib/Analysis/InnerProductSpace/Basic
import Mathlib/Topology/Instances.Real

/-!
Ordered eigenvalues and Lipschitz (P1) for self-adjoint operators (finite-dim).

We provide a Prop-level interface capturing the 1‑Lipschitz property of the top
two ordered eigenvalues under operator-norm perturbations. This file is a
staging point: replace the Prop-level `OrderedEigenLipschitz` with a concrete
statement via Weyl/min–max when the mathlib API needed is available.
-/

namespace YM

variable {𝕂 : Type*} [IsROrC 𝕂]
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]
variable [FiniteDimensional 𝕂 E]

/-- Real Rayleigh quotient for a self-adjoint operator on a unit vector. -/
def rayleigh (T : E →L[𝕂] E) (x : E) : ℝ :=
  realPart (⟪x, T x⟫_𝕂) / ‖x‖^2

/-- The unit sphere in `E`. -/
def unitSphere : Set E := { x | ‖x‖ = 1 }

/-- Top ordered eigenvalue functional defined as the supremum of the Rayleigh
quotient over the unit sphere (Courant–Fischer). -/
def lambda₁ (T : E →L[𝕂] E) : ℝ :=
  sSup (rayleigh T '' unitSphere)

/-- Second ordered eigenvalue functional via an inf-sup recipe (schematic). -/
def lambda₂ (T : E →L[𝕂] E) : ℝ :=
  sInf (sSup ''
    { S : Set E | ∃ v ∈ unitSphere, S = { x | ‖x‖ = 1 ∧ ⟪x, v⟫_𝕂 = 0 } |> rayleigh T '' })

/-- Interface: the top two ordered eigenvalue functionals are 1‑Lipschitz in
operator norm on self-adjoint operators (finite dimension). -/
def OrderedEigenLipschitz
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ) : Prop :=
  ∀ ⦃X Y : E →L[𝕂] E⦄,
    IsSelfAdjoint X → IsSelfAdjoint Y →
    |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖

/-- Staging theorem: expose the P1 hypothesis in a convenient form for
`ym/SpectralStability.lean`. Replace this with a concrete construction of
`λ₁, λ₂` and a proof via Weyl/min–max when ready. -/
theorem P1_expose
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ)
    (hP1 : OrderedEigenLipschitz (𝕂 := 𝕂) (E := E) λ₁ λ₂) :
    ∀ {X Y : E →L[𝕂] E},
      IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖ :=
  by intro X Y hX hY; exact hP1 hX hY

/-- Blocker note (to be removed when concrete proof is added):
Proving the Lipschitz property for `lambda₁, lambda₂` as defined above requires
min–max (Courant–Fischer) and Weyl-type inequalities in mathlib. Once available,
replace `OrderedEigenLipschitz` usage with a concrete lemma
`ordered_eigen_lipschitz_for_lambda12`. -/

end YM
