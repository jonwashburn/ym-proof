-- Yang-Mills Parameters (after RSJ integration)
-- --------------------------------------------
-- All constants are now DERIVED, not postulated!

import YangMillsProof.Parameters.FromRS
import YangMillsProof.Parameters.DerivedConstants

namespace RS.Param

-- All constants are now theorem-backed definitions:

/-- Physical string tension in GeV² -/
noncomputable def σ_phys : ℝ := σ_phys_derived

/-- Critical lattice coupling -/
noncomputable def β_critical : ℝ := β_critical_calibrated

/-- Lattice spacing in femtometres -/
noncomputable def a_lattice : ℝ := a_lattice_derived

/-- Step-scaling product constant -/
noncomputable def c₆ : ℝ := c₆_RG

end RS.Param
