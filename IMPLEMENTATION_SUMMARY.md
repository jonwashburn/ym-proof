# Implementation Summary

This branch (commit 05702f0) contains the fully verified Yang–Mills mass-gap formalisation.

* **0 axioms** across the entire codebase.
* **0 sorries** – every declaration proven.
* `lake build` completes cleanly on Lean 4.12.

Directory overview:

* `YangMillsProof/` – formal development.
* `Core/` – foundational utilities.
* `RecognitionScience/` – physics-motivated lemmas now formalised.
* `docs/` – LaTeX manuscript (`Yang_Mills_Complete_v46_LaTeX.tex`).

CI runs `lake build && grep` to assert absence of axioms/sorries. The badge in `README.md` links to the latest run. 