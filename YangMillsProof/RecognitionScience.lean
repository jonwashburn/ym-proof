/-
  Recognition Science: Clean Foundation Import
  ===========================================

  This module imports the clean, zero-axiom, zero-sorry foundation
  from ledger-foundation instead of the broken local implementation.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

-- Import the clean foundation
import YangMillsProof.foundation_clean.RecognitionScience
import YangMillsProof.foundation_clean.MinimalFoundation

namespace RecognitionScience

-- Re-export everything from the clean foundation
export RecognitionScience.Minimal

end RecognitionScience
