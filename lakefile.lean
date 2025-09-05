import Lake
open Lake DSL
package «ym-proof»
require mathlib from git
  https://github.com/leanprover-community/mathlib4.git
@[default_target]
lean_lib YM where
  roots := #[ym]
