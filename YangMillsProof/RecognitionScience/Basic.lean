/-
  Recognition Science Basic Definitions
  ====================================

  Core types and structures for the RS framework.

  Author: Jonathan Washburn
-/

import Mathlib.LinearAlgebra.Matrix.SpecialLinearGroup
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.Basic

namespace RecognitionScience

open Complex

/-- The gauge group SU(3) -/
abbrev SU3 := Matrix.SpecialUnitaryGroup (Fin 3) ℂ

/-- A site on the 4D hypercubic lattice -/
structure Site where
  x : Fin 4 → ℤ  -- 4D coordinates

/-- A link between two sites -/
structure Link where
  source : Site
  target : Site
  direction : Fin 4
  h_adjacent : target.x = Function.update source.x direction (source.x direction + 1)

/-- A plaquette (elementary square) on the lattice -/
structure Plaquette where
  corner : Site
  dir1 : Fin 4
  dir2 : Fin 4
  h_distinct : dir1 ≠ dir2

/-- A surface is a collection of plaquettes -/
def Surface : Type := Finset Plaquette

/-- The gauge field assigns an SU(3) element to each link -/
def GaugeField := Link → SU3

/-- The four links around a plaquette -/
def plaquette_links (P : Plaquette) : List Link :=
  let x := P.corner
  let i := P.dir1
  let j := P.dir2
  -- Link 1: x → x + e_i
  let link1 : Link := ⟨x, ⟨Function.update x.x i (x.x i + 1)⟩, i, rfl⟩
  -- Link 2: x + e_i → x + e_i + e_j
  let x_plus_i : Site := ⟨Function.update x.x i (x.x i + 1)⟩
  let link2 : Link := ⟨x_plus_i, ⟨Function.update x_plus_i.x j (x_plus_i.x j + 1)⟩, j, rfl⟩
  -- Link 3: x + e_j → x + e_i + e_j (reversed)
  let x_plus_j : Site := ⟨Function.update x.x j (x.x j + 1)⟩
  let x_plus_ij : Site := ⟨Function.update x_plus_i.x j (x_plus_i.x j + 1)⟩
  let link3 : Link := ⟨x_plus_ij, x_plus_j, i, by
    simp [x_plus_ij, x_plus_j, x_plus_i]
    funext k
    by_cases h1 : k = i
    · simp [h1, Function.update_same]
      by_cases h2 : k = j
      · simp [h2] at h1
        exact absurd h1 P.h_distinct
      · simp [h2, Function.update_noteq]
    · by_cases h2 : k = j
      · simp [h1, h2, Function.update_noteq, Function.update_same]
      · simp [h1, h2, Function.update_noteq]⟩
  -- Link 4: x → x + e_j (reversed)
  let link4 : Link := ⟨x_plus_j, x, j, by
    simp [x_plus_j]
    funext k
    by_cases h : k = j
    · simp [h, Function.update_same]
    · simp [h, Function.update_noteq]⟩
  [link1, link2, link3, link4]

/-- Holonomy around a plaquette -/
noncomputable def gauge_holonomy (U : GaugeField) (P : Plaquette) : SU3 :=
  -- Compute the ordered product U₁ U₂ U₃⁻¹ U₄⁻¹
  let links := plaquette_links P
  match links with
  | [l1, l2, l3, l4] => U l1 * U l2 * (U l3)⁻¹ * (U l4)⁻¹
  | _ => 1  -- Should never happen, but needed for exhaustiveness

/-- Area of a surface (number of plaquettes) -/
def area (Σ : Surface) : ℕ := Σ.card

/-- Recognition Science quantum unit -/
def rsQuantum : ℕ := 146

/-- Half quantum (fundamental excitation) -/
def halfQuantum : ℕ := 73

/-- Verify the relationship -/
lemma half_quantum_correct : 2 * halfQuantum = rsQuantum := by
  unfold halfQuantum rsQuantum
  norm_num

end RecognitionScience
