/-
  Lattice Structure for 4D Gauge Theory
  =====================================

  Defines sites, directions, and basic operations on the 4D hypercubic lattice.
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.ZMod.Basic

namespace YangMillsProof.Gauge

/-- A site on the 4D hypercubic lattice -/
def Site := Fin 4 → ℤ

/-- Direction on the lattice (μ = 0,1,2,3) -/
abbrev Dir := Fin 4

/-- Shift a site by one unit in direction μ -/
def Site.shift (x : Site) (μ : Dir) : Site :=
  fun ν => if ν = μ then x ν + 1 else x ν

-- Notation for site shift
notation:max x " + " μ:max => Site.shift x μ

/-- A link is specified by a site and a direction -/
structure Link where
  site : Site
  dir : Dir

/-- A plaquette is specified by a site and two different directions -/
structure Plaquette where
  site : Site
  dir1 : Dir
  dir2 : Dir
  ne : dir1 ≠ dir2

/-- The four links forming a plaquette boundary -/
def Plaquette.links (P : Plaquette) : Fin 4 → Link :=
  fun i => match i with
  | 0 => ⟨P.site, P.dir1⟩
  | 1 => ⟨P.site + P.dir1, P.dir2⟩
  | 2 => ⟨P.site + P.dir2, P.dir1⟩
  | 3 => ⟨P.site, P.dir2⟩

/-- Orientation of links in plaquette (forward = true, backward = false) -/
def Plaquette.orientation : Fin 4 → Bool :=
  fun i => match i with
  | 0 => true   -- forward
  | 1 => true   -- forward
  | 2 => false  -- backward
  | 3 => false  -- backward

end YangMillsProof.Gauge
