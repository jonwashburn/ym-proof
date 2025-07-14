/-
  Recognition Science Parameters (Compatibility Module)
  ====================================================

  This module re-exports all Recognition Science parameters from the split
  Definitions and Bounds modules for backward compatibility.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

-- Import both modules to provide all symbols
import Parameters.Definitions
import Parameters.Bounds

-- Re-export everything from the RS.Param namespace
-- This ensures backward compatibility for existing imports
namespace RS.Param

-- All definitions and theorems are automatically available
-- through the imports above since they're in the same namespace

end RS.Param
