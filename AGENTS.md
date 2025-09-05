### Agents and Roles

- Use branches and file ownership per guardrails below.

### Guardrails for Parallel Agents (Round 1)

- Interface freeze: Do not modify shared core types (e.g., TransferKernel) in this round.
- Step 3 must implement its adapter in a new file (e.g., ym/Adapter/MatrixToTransfer.lean) without editing ym/Transfer.lean.
- One branch per agent; change only owned files.

### Branch and file ownership

- Step 1 (P1 Eigenvalue Lipschitz):
  - Branch: feat/p1-eigen-order
  - Files: ym/EigenvalueOrder.lean (or a new section in ym/SpectralStability.lean)

- Step 2 (PF 3×3 gap):
  - Branch: feat/pf3x3
  - Files: ym/PF3x3.lean; example in ym/Examples.lean

- Step 3 (Matrix→Transfer adapter):
  - Branch: feat/adapter-matrix-transfer
  - Files: ym/Adapter/MatrixToTransfer.lean (new); do not edit ym/Transfer.lean

- Step 4 (Embedding + convergence):
  - Branch: feat/embedding
  - Files: ym/Embedding.lean (or extend ym/Continuum.lean)

- Step 5 (OS→PF uniform γ):
  - Branch: feat/os-to-pf
  - Files: ym/Reflection.lean; new lemmas in ym/Transfer.lean without changing existing signatures

### PR acceptance checklist (each agent)

- lake build passes; no new axioms or sorries.
- Only owned files changed.
- If a shared API change is required, open a tiny interface PR first, then rebase.
- On missing mathlib lemmas, log a one-line blocker and stop.
