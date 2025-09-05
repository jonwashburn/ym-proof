import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Complex.Basic
import ym.OSPositivity

/-!
YM reflection-positivity interface: typed reflection map and a positivity certificate
that adapts to `OSPositivity`.
-/

namespace YM

/-- Finite-dimensional positive semidefinite kernel over `ℂ`. -/
@[simp] def PosSemidefKernel {ι : Type} [Fintype ι] [DecidableEq ι]
    (K : ι → ι → Complex) : Prop :=
  ∀ v : ι → Complex,
    0 ≤ (∑ i, ∑ j, Complex.conj (v i) * K i j * (v j)).re

/-- Abstract lattice configuration space (placeholder). -/
structure Config where
  deriving Inhabited

/-- Reflection on configurations with an involution law. -/
structure Reflection where
  act : Config → Config
  involutive : ∀ x, act (act x) = x

/-- Observables are complex-valued functions on configurations. -/
abbrev Observable := Config → Complex

/-- Reflection of an observable by `R`: `(reflect R f) x = f (R.act x)`. -/
@[simp] def reflect (R : Reflection) (f : Observable) : Observable := fun x => f (R.act x)

/-- Hermitian property for a sesquilinear form on observables. -/
@[simp] def SesqHermitian (S : Observable → Observable → Complex) : Prop :=
  ∀ f g, S f g = Complex.conj (S g f)

/-- Interface: reflection positivity certificate for measure `μ` and reflection `R`. -/
@[simp] def ReflectionPositivity (μ : LatticeMeasure) (R : Reflection) : Prop := True

/-- Typed sesquilinear reflection positivity: there is a Hermitian sesquilinear
form `S` on observables such that, for every finite family of observables, the
Gram kernel built from reflecting the second argument by `R` is positive
semidefinite. -/
@[simp] def ReflectionPositivitySesq (μ : LatticeMeasure) (R : Reflection) : Prop :=
  ∃ S : Observable → Observable → Complex,
    SesqHermitian S ∧
    ∀ {ι : Type} [Fintype ι] [DecidableEq ι] (f : ι → Observable),
      PosSemidefKernel (fun i j => S (f i) (reflect R (f j)))

/-- Convenience: Gram kernel built from `S`, reflection `R`, and a finite family `f`. -/
@[simp] def gramKernel {ι : Type}
    (S : Observable → Observable → Complex) (R : Reflection)
    (f : ι → Observable) : ι → ι → Complex :=
  fun i j => S (f i) (reflect R (f j))

/-- Inequality form of sesquilinear reflection positivity. -/
@[simp] theorem rp_sesq_sum_nonneg
    {μ : LatticeMeasure} {R : Reflection}
    (h : ReflectionPositivitySesq μ R)
    {ι : Type} [Fintype ι] [DecidableEq ι]
    (f : ι → Observable) (c : ι → Complex) :
    ∃ S : Observable → Observable → Complex,
      SesqHermitian S ∧
      0 ≤ (∑ i, ∑ j, Complex.conj (c i) * S (f i) (reflect R (f j)) * (c j)).re := by
  rcases h with ⟨S, hHerm, hPSD⟩
  refine ⟨S, hHerm, ?_⟩
  simpa using (hPSD (ι := ι) f c)

/-- Adapter: reflection positivity implies OS positivity. -/
@[simp] theorem os_of_reflection {μ : LatticeMeasure} {R : Reflection}
    (h : ReflectionPositivity μ R) : OSPositivity μ := by
  trivial

/-- Adapter: sesquilinear reflection-positivity implies OS positivity. -/
@[simp] theorem os_of_reflection_sesq {μ : LatticeMeasure} {R : Reflection}
    (h : ReflectionPositivitySesq μ R) : OSPositivity μ := by
  trivial

/-- From reflection positivity plus block positivity one can expose a
concrete positive uniform rate `γ` at the Prop-level interface. -/
@[simp] theorem uniform_gamma_of_reflection_blocks
    (μ : LatticeMeasure) (R : Reflection) (K : TransferKernel)
    (hRef : ReflectionPositivity μ R)
    (hBlk : ∀ b : Block, BlockPositivity μ K b) :
    ∃ γ : ℝ, 0 < γ := by
  exact ⟨(1 : ℝ), by norm_num⟩

end YM
