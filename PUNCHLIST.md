## Punch List - Next Tasks

- Replace `sorry` in `RSImport.BasicDefinitions` (`tsum_eq_zero_iff_all_eq_zero` lemma)
- Complete `activity_zero_iff_vacuum` proof in `Stage0_RS_Foundation/ActivityCost.lean`
- Add Stage 0 modules to `lakefile` roots and ensure `lake build` succeeds
- Incrementally add Stage 1–2 roots; fix any sorries before committing
- Add CI step: fail build if any sorries or axioms in current roots
- Update README to reflect current sorry status and add quick-start cache commands
- Reconcile `REPOSITORY_LOCK.md` language with actual build status or rename file
- Compress CI caches with zstd for reduced storage/bandwidth
- Integrate community mathlib cache (`lake exe cache get/put`) for first-clone speed-up
- Provide Dockerfile for hermetic Lean+mathlib build environment 