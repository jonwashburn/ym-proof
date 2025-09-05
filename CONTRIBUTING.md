# Contributing

This project uses parallel agent branches with strict guardrails to avoid collisions.

## Quick start

- Branch from `main`.
- One branch per task (e.g., `feat/pf3x3`, `feat/embedding`).
- Edit only the files listed for your task in `AGENTS.md`.
- If you must change a shared interface (e.g., core types), open a tiny “interface PR” first; others rebase after it merges.

## Guardrails

- No new axioms or sorries.
- Preserve existing public signatures unless explicitly approved.
- If a needed mathlib lemma is missing, log a one‑line blocker in the PR and stop.

## Tasks and ownership

See `AGENTS.md` for the current task matrix, branch names, and file ownership.

## Building locally



## Pipeline usage examples

Open `ym/Pipeline.lean` in the `YM.Examples` namespace:

- `YM.Examples.toy_end_to_end` (γ = 1)
- `YM.Examples.two_thirds_end_to_end` (γ = 2/3)
- `YM.Examples.three_fourths_end_to_end` (γ = 3/4)

You can also use the higher-level export from `ym/Interfaces.lean`:

Miscellaneous:
  --help -h          display this message
  --features -f      display features compiler provides (eg. LLVM support)
  --version -v       display version number
  --githash          display the git commit hash number used to build this binary
  --run              call the 'main' definition in a file with the remaining arguments
  --o=oname -o       create olean file
  --i=iname -i       create ilean file
  --c=fname -c       name of the C output file
  --bc=fname -b      name of the LLVM bitcode file
  --stdin            take input from stdin
  --root=dir         set package root directory from which the module name of the input file is calculated
                     (default: current working directory)
  --trust=num -t     trust level (default: max) 0 means do not trust any macro,
                     and type check all imported modules
  --quiet -q         do not print verbose messages
  --memory=num -M    maximum amount of memory that should be used by Lean
                     (in megabytes)
  --timeout=num -T   maximum number of memory allocations per task
                     this is a deterministic way of interrupting long running tasks
  --threads=num -j   number of threads used to process lean files
  --tstack=num -s    thread stack size in Kb
  --server           start lean in server mode
  --worker           start lean in server-worker mode
  --plugin=file      load and initialize Lean shared library for registering linters etc.
  --load-dynlib=file load shared library to make its symbols available to the interpreter
  --json             report Lean output (e.g., messages) as JSON (one per line)
  --deps             just print dependencies of a Lean input
  --print-prefix     print the installation prefix for Lean and exit
  --print-libdir     print the installation directory for Lean's built-in libraries and exit
  --profile          display elaboration/type checking time for each definition/theorem
  --stats            display environment statistics
  -D name=value      set a configuration option (see set_option command)p

## Embedding demo

When `feat/embedding` lands, a small example invoking
`gap_persists_via_embedding` will show how to define embeddings
`ι n : F n →L E` and verify norm convergence.

## CI

PRs must:
- build with `lake build`
- pass the axiom/sorry grep checks
- keep example `#check`s compiling
