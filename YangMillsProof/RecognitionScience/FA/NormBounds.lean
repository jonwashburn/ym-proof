/-
  Recognition Science Functional Analysis Norm Bounds
  ==================================================

  This module proves L² norm bounds for gauge theory observables
  in the Recognition Science framework.

  Author: Jonathan Washburn
-/

import YangMillsProof.RecognitionScience.Basic
import YangMillsProof.Core.Constants

namespace RecognitionScience.FA

open YangMillsProof

/-- L² bound for gauge observables -/
theorem l2_bound : ∀ (f : GaugeLedgerState → ℝ),
    (∀ s : GaugeLedgerState, |f s| ≤ stateCost s) →
    ∑' s : GaugeLedgerState, (f s)^2 * gibbs_weight s < ∞ := by
  intro f h_bound

  -- The L² norm is finite because:
  -- 1. f is bounded by stateCost
  -- 2. The Gibbs weights exp(-βE) decay exponentially
  -- 3. The state space has polynomial growth

  -- Key estimate: ∑ |f(s)|² exp(-βE_s) ≤ ∑ E_s² exp(-βE_s)
  -- The right side converges by:
  -- - Polynomial growth: #{s : E_s ≤ E} ~ E^d
  -- - Exponential decay beats polynomial growth

  -- Use Cauchy-Schwarz: ∑ |f|² w ≤ (∑ |f|⁴ w)^(1/2) * (∑ w)^(1/2)
  -- But simpler: use |f| ≤ E and exponential decay

  -- ∑ f² exp(-βE) ≤ ∑ E² exp(-βE)
  -- Split by energy shells: E ∈ [n, n+1)
  -- Each shell has ≤ C(n+1)^d states (polynomial growth)
  -- Contribution: ≤ C(n+1)^d * n² * exp(-βn)
  -- Sum over n: ∑ n^(d+2) exp(-βn) < ∞ by ratio test

  -- Requires summable_exp_gap from TransferMatrix
  apply l2_summable_from_exp_gap f h_bound

/-- Stronger bound: exponential decay -/
theorem exponential_bound (f : GaugeLedgerState → ℝ) :
    (∀ s : GaugeLedgerState, |f s| ≤ stateCost s) →
    ∃ C α : ℝ, C > 0 ∧ α > 0 ∧
    ∀ s : GaugeLedgerState, |f s| * gibbs_weight s ≤ C * Real.exp (-α * stateCost s) := by
  intro h_bound

  -- Choose α slightly less than β to ensure convergence
  use 1, temperature * massGap / 2

  constructor
  · norm_num
  constructor
  · apply mul_pos temperature_pos
    exact div_pos massGap_pos (by norm_num : (0 : ℝ) < 2)

  intro s
  -- |f(s)| ≤ E_s and gibbs_weight s = exp(-β E_s)
  -- So |f(s)| * gibbs_weight s ≤ E_s * exp(-β E_s)
  -- This is bounded by C * exp(-α E_s) for suitable C, α < β

  -- Function g(x) = x * exp(-βx) has maximum at x = 1/β
  -- Max value is (1/β) * exp(-1) < 1/β

  -- For our case: E * exp(-βE) ≤ 1/β
  -- With β = 1/temperature and α = β/2:
  -- E * exp(-βE) = E * exp(-αE) * exp(-αE)
  --              ≤ (1/α) * exp(-αE) using max of x*exp(-αx)
  --              = (2/β) * exp(-αE)

  calc |f s| * gibbs_weight s
    ≤ stateCost s * gibbs_weight s := by
      apply mul_le_mul_of_nonneg_right (h_bound s)
      exact gibbs_weight_nonneg s
    _ = stateCost s * Real.exp (-temperature⁻¹ * stateCost s) := by
      unfold gibbs_weight
      rfl
    _ ≤ 1 * Real.exp (-(temperature * massGap / 2) * stateCost s) := by
      -- Use that x * exp(-x/T) ≤ T for all x ≥ 0
      -- Here x = stateCost s, T = temperature
      -- Requires derivative of x * exp(-x)

      -- The function g(x) = x * exp(-x/T) has derivative
      -- g'(x) = exp(-x/T) - (x/T) * exp(-x/T) = exp(-x/T)(1 - x/T)
      -- Critical point at x = T, with g(T) = T * exp(-1) < T
      -- Since g(0) = 0 and g(∞) = 0, the max is at x = T

      -- For our bound with α = β/2 = 1/(2T):
      -- x * exp(-x/T) = x * exp(-αx) * exp(-αx)
      -- Using h(x) = x * exp(-αx) with max at x = 1/α = 2T:
      -- h(1/α) = (1/α) * exp(-1) = 2T/e

      -- Therefore: x * exp(-x/T) ≤ (2T/e) * exp(-αx) ≤ 1 * exp(-αx)
      -- since 2T/e < T < 1 when T = temperature < 1/2

      apply max_x_exp_neg_x (stateCost s) (temperature * massGap / 2)
      · exact stateCost_nonneg s
      · apply mul_pos temperature_pos
        exact div_pos massGap_pos (by norm_num : (0 : ℝ) < 2)

end RecognitionScience.FA
