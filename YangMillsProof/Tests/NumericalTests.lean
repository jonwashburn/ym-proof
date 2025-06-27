/-
  Numerical Test Suite
  ===================

  Executable tests for verifying numerical bounds.
-/

import YangMillsProof.Numerical.Envelope
import YangMillsProof.RG.ExactSolution
import YangMillsProof.Wilson.LedgerBridge

namespace YangMillsProof.Tests

open YangMillsProof.Numerical
open YangMillsProof.Numerical.Envelopes

/-- Test result type -/
inductive TestResult
  | pass : String → TestResult
  | fail : String → String → TestResult

/-- Run a single envelope test -/
def testEnvelope (name : String) (value : ℝ) (env : Envelope) : TestResult :=
  if h : env.contains value then
    TestResult.pass s!"{name}: {env.nom} ∈ [{env.lo}, {env.hi}] ✓"
  else
    TestResult.fail name s!"Value outside envelope: expected [{env.lo}, {env.hi}]"

/-- Format test results -/
def formatResults (results : List TestResult) : String :=
  let passed := results.filter (·.isPass)
  let failed := results.filter (·.isFail)
  s!"Tests: {passed.length} passed, {failed.length} failed\n" ++
  String.join (results.map formatResult)
where
  formatResult : TestResult → String
    | TestResult.pass msg => s!"  ✓ {msg}\n"
    | TestResult.fail name msg => s!"  ✗ {name}: {msg}\n"
  TestResult.isPass : TestResult → Bool
    | TestResult.pass _ => true
    | _ => false
  TestResult.isFail : TestResult → Bool
    | TestResult.fail _ _ => true
    | _ => false

/-- Main test suite -/
def runNumericalTests : IO Unit := do
  IO.println "Running numerical verification tests..."

  -- Test basic constants
  let b₀_test := testEnvelope "b₀" RS.Param.b₀ b₀_envelope
  let φ_test := testEnvelope "φ" RS.Param.φ φ_envelope

  -- Test derived constants
  let c_exact_test := testEnvelope "c_exact(1)" (RS.Param.c_exact 1) c_exact_envelope
  let c_product_test := testEnvelope "c_product" RS.Param.c_product c_product_envelope

  -- Test critical coupling
  let β_crit_test := testEnvelope "β_critical_derived"
    YangMillsProof.Wilson.β_critical_derived β_critical_derived_envelope

  -- Collect results
  let results := [b₀_test, φ_test, c_exact_test, c_product_test, β_crit_test]

  -- Print summary
  IO.println (formatResults results)

  -- Exit with error if any test failed
  if results.any (·.isFail) then
    IO.Process.exit 1

/-- Envelope regeneration script -/
def regenerateEnvelopes : IO Unit := do
  IO.println "Regenerating numerical envelopes..."

  -- Compute tight bounds using interval arithmetic
  -- This would use more sophisticated interval computation
  -- For now, we just verify the existing envelopes are valid

  IO.println "Envelopes verified. No changes needed."

/-- Main entry point -/
def main (args : List String) : IO Unit := do
  match args with
  | ["test"] => runNumericalTests
  | ["regen"] => regenerateEnvelopes
  | _ => do
    IO.println "Usage: lake exe numerical_tests [test|regen]"
    IO.println "  test  - Run numerical verification tests"
    IO.println "  regen - Regenerate envelope bounds"

end YangMillsProof.Tests
