import Mathlib.Data.Real.Basic

namespace YangMillsProof.Stage1_GaugeEmbedding

/-- Lattice spacing parameter -/
structure LatticeScale where
  a : ℝ
  a_pos : 0 < a

/-- Extended voxel face with lattice scale -/
structure ScaledVoxelFace (scale : LatticeScale) where
  x : Fin 4 → ℤ  -- 4D position
  μ : Fin 4       -- direction
  ν : Fin 4       -- perpendicular direction
  h_neq : μ ≠ ν

end YangMillsProof.Stage1_GaugeEmbedding
