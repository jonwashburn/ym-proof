import Lake
open Lake DSL

package YangMillsProof where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

require RecognitionScience from git
  "https://github.com/jonwashburn/recognition-ledger.git" @ "main"

@[default_target]
lean_lib YangMillsProof where
  roots := #[`YangMillsProof]
