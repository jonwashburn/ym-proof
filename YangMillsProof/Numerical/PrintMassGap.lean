import YangMillsProof.Parameters.Assumptions

open RS.Param

/-!
  # Mass gap sanity check

  Run with `lake exec printMassGap` after adding the following target to `lakefile.lean`:

  ````lean
  lean_exe printMassGap where
    root := `YangMillsProof.Numerical.PrintMassGap
  ````
-/

#eval IO.println s!"massGap = {massGap}"
