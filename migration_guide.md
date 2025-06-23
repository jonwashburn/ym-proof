# Yang-Mills Proof Migration Guide

## Overview
This guide helps migrate the existing proof files to the new stage-based structure aligned with the adjusted roadmap.

## File Mapping

### Stage 0: RS Foundation
- `RSImport/BasicDefinitions.lean` → Keep in place, but:
  - Extract activity cost to `Stage0_RS_Foundation/ActivityCost.lean`
  - Replace the `sorry` in `cost_zero_iff_vacuum` using new theorems
- `RSImport/GoldenRatio.lean` → Keep as utility file

### Stage 1: Gauge Embedding  
- `GaugeResidue.lean` → Split into:
  - Core definitions → `Stage1_GaugeEmbedding/GaugeStructures.lean`
  - Embedding logic → `Stage1_GaugeEmbedding/GaugeToLedger.lean`
  - Cost bounds → Keep for Stage 2

### Stage 2: Lattice Theory
- `TransferMatrix.lean` → Move to `Stage2_LatticeTheory/TransferMatrixGap.lean`
  - Remove hard-coded `massGap = 1.11`
  - Import from `Infrastructure/PhysicalConstants.lean` instead

### Stage 3: OS Reconstruction
- `OSReconstruction.lean` → Split into:
  - Reflection positivity → `Stage3_OSReconstruction/ReflectionPositivity.lean`
  - Field construction → `Stage3_OSReconstruction/ContinuumReconstruction.lean`
  - Add new `FractionalActionRP.lean` for fractional powers

### Stage 4: Continuum Limit
- Extract continuum limit logic from `OSReconstruction.lean`
- Create `Stage4_ContinuumLimit/MassGapPersistence.lean`

### Stage 5: Renormalization
- New content based on power counting arguments
- Already created `Stage5_Renormalization/IrrelevantOperator.lean`

### Stage 6: Main Theorem
- `Complete.lean` → Rewrite as `Stage6_MainTheorem/Complete.lean`
- Import all stages and tie together

## Import Updates

Replace imports like:
```lean
import YangMillsProof.GaugeResidue
```

With stage-specific imports:
```lean
import YangMillsProof.Stage1_GaugeEmbedding.GaugeStructures
import YangMillsProof.Infrastructure.PhysicalConstants
```

## Testing Migration

1. Copy files to new locations
2. Update imports in each file
3. Run `lake build` on individual stages:
   ```bash
   lake build YangMillsProof.Stage0_RS_Foundation
   lake build YangMillsProof.Stage1_GaugeEmbedding
   # etc.
   ```

## Cleanup
Once migration is complete:
1. Remove old files from root `YangMillsProof/` directory
2. Update `lakefile.lean` to use `lakefile_new.lean`
3. Run full build: `lake build` 