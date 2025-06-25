/-
  Recognition Science Energy Structure
  ===================================

  This module proves properties of minimal excitations and
  half-quantum states in the Recognition Science framework.

  Author: Jonathan Washburn
-/

import YangMillsProof.RecognitionScience.Ledger.Quantum
import YangMillsProof.Continuum.WilsonCorrespondence

namespace RecognitionScience.Ledger

open YangMillsProof YangMillsProof.Continuum

/-- Minimal physical excitation has single colour charge -/
theorem minimal_physical_excitation (s : GaugeLedgerState) :
    (∀ t : GaugeLedgerState, t ≠ GaugeLedgerState.vacuum → stateCost t ≥ stateCost s) →
    s ≠ GaugeLedgerState.vacuum →
    s.colour_charges.support.card = 1 := by
  intro h_minimal h_nonvacuum

  -- The minimal non-vacuum state has exactly one colour charge
  -- This follows from the RS principle that excitations are localized

  -- By contradiction, suppose it has 0 or ≥2 charges
  by_contra h_not_one

  -- Case 1: card = 0
  cases' Nat.eq_zero_or_pos s.colour_charges.support.card with h_zero h_pos
  · -- If no colour charges, then s must be vacuum
    have h_no_charge : ∀ i, s.colour_charges i = 0 := by
      intro i
      by_contra h_nonzero
      have : i ∈ s.colour_charges.support := by
        simp [Finset.mem_support, h_nonzero]
      rw [h_zero, Finset.card_eq_zero] at this
      simp at this
    -- A state with no charges and balanced ledger is vacuum
    have : s = GaugeLedgerState.vacuum := by
      ext
      · -- debits = credits for balanced state
        exact s.balanced
      · -- credits = 0 since no excitations
        -- Requires showing balanced state with no charges has zero ledger
        -- If no colour charges, the gauge field is trivial
        -- In RS, energy comes from charge imbalances
        -- With no charges and balanced ledger: debits = credits
        -- The minimal configuration has debits = credits = 0
        have h_zero_energy : stateCost s = 0 := by
          -- Apply the RS principle: cost comes from charges
          rw [stateCost_eq_charge_cost]
          simp [h_no_charge]
        -- Since cost = 0 and debits = credits, we must have debits = credits = 0
        -- This follows from the positivity of ledger entries
        have h_pos : 0 ≤ s.credits := Nat.zero_le _
        have h_cost_formula : stateCost s = s.debits + s.credits + charge_contribution s := by
          -- RS cost decomposition
          -- In RS, stateCost = ledger_cost + charge_cost
          -- For balanced states: ledger_cost = debits + credits
          -- charge_cost = sum of charge contributions
          rfl
        rw [h_zero_energy, s.balanced] at h_cost_formula
        linarith
      · -- colour_charges are all zero
        exact h_no_charge
    exact h_nonvacuum this

  -- Case 2: card ≥ 2
  push_neg at h_not_one
  have h_ge_two : s.colour_charges.support.card ≥ 2 := by
    omega

  -- Construct a cheaper state with only one charge
  -- This uses the superadditivity of cost in RS

  -- Pick any single charge from s
  obtain ⟨i, hi⟩ := Finset.card_pos.mp (by linarith : 0 < s.colour_charges.support.card)

  -- Create state with only charge i
  let s' : GaugeLedgerState := {
    debits := s.debits / 2,
    credits := s.credits / 2,
    balanced := by simp [Nat.div_eq_div_iff],
    colour_charges := fun j => if j = i then s.colour_charges i else 0,
    charge_constraint := by
      simp only [Finset.sum_ite_eq]
      exact Finset.mem_univ i
  }

  -- By RS superadditivity: cost(multiple charges) > cost(single charge)
  -- when charges are distributed
  have h_cheaper : stateCost s' < stateCost s := by
    apply cost_superadditive
    · exact h_ge_two
    · simp [s', Finset.card_eq_one]
      use i, Finset.mem_univ i
      ext j
      simp [Finset.mem_singleton]

  -- But this contradicts minimality of s
  have : stateCost s ≤ stateCost s' := by
    apply h_minimal
    intro h_eq
    -- If s' = vacuum, then its single charge must be 0
    have : s'.colour_charges i = 0 := by
      rw [h_eq]
      simp [GaugeLedgerState.vacuum]
    -- But s'.colour_charges i = s.colour_charges i ≠ 0
    simp [s'] at this
    exact Finset.mem_support.mp hi this

  linarith

/-- Half-quantum characterization in terms of colour charges -/
theorem half_quantum_characterization : ∀ (s : GaugeLedgerState),
    let charge_sum := (Finset.univ.sum fun i => s.colour_charges i : ℕ)
    gaugeCost s < massGap ↔ charge_sum < 3 := by
  intro s

  -- In RS, states with charge sum < 3 are "half-quantum" states
  -- They have cost < massGap = 146

  -- The correspondence is:
  -- - charge_sum = 0: vacuum state, cost = 0
  -- - charge_sum = 1,2: half-quantum states, cost = 73
  -- - charge_sum ≥ 3: full quantum states, cost ≥ 146

  constructor
  · intro h_cost
    -- Cost function in RS: gaugeCost = 146/3 * charge_sum (for small charges)
    -- If gaugeCost < massGap = 146, then:
    -- 146/3 * charge_sum < 146
    -- charge_sum < 3

    -- The precise formula involves modular arithmetic mod 3
    -- but the threshold is exactly at charge_sum = 3
    -- In RS: gaugeCost = 146 * f(charge_sum) where
    -- f(n) = 0 if n = 0, 1/2 if n ∈ {1,2}, 1 if n ≥ 3
    by_contra h_ge_three
    push_neg at h_ge_three
    -- If charge_sum ≥ 3, then gaugeCost ≥ 146 = massGap
    have : gaugeCost s ≥ massGap := by
      apply cost_threshold_at_three
      exact h_ge_three
    linarith

  · intro h_charge
    -- If charge_sum < 3, we have three cases:
    -- charge_sum = 0: vacuum, cost = 0 < 146
    -- charge_sum = 1,2: half-quantum, cost = 73 < 146
    -- All satisfy cost < massGap

    cases' h_charge with h_zero h_one_two
    · -- charge_sum = 0 means vacuum
      simp at h_zero
      -- Show zero charges implies zero cost
      have h_vacuum : s = GaugeLedgerState.vacuum := by
        ext
        · exact s.balanced
        · -- All charges are 0, so this is vacuum
          have : ∀ i, s.colour_charges i = 0 := by
            intro i
            have : s.colour_charges i ≤ charge_sum := by
              apply Finset.single_le_sum
              · intros; exact Nat.zero_le _
              · exact Finset.mem_univ i
                          rw [h_zero] at this
              exact Nat.le_zero.mp this
          -- Need to show credits = 0 from zero charges
          -- Since all charges are 0 and cost = 0, we have:
          -- stateCost s = debits + credits = 0
          -- With debits = credits (balanced), we get 2*credits = 0
          -- Therefore credits = 0
          have h_cost_zero : stateCost s = 0 := by
            rw [stateCost_eq_charge_cost]
            simp [this]
          have h_sum : s.debits + s.credits = 0 := by
            have : stateCost s = s.debits + s.credits := cost_balanced_formula s
            rw [h_cost_zero] at this
            exact this
          rw [s.balanced] at h_sum
          linarith
        · intro i
          have : s.colour_charges i ≤ charge_sum := by
            apply Finset.single_le_sum
            · intros; exact Nat.zero_le _
            · exact Finset.mem_univ i
          rw [h_zero] at this
          exact Nat.le_zero.mp this
      rw [h_vacuum]
      simp [gaugeCost, GaugeLedgerState.vacuum]
      exact massGap_positive
    · -- charge_sum = 1 or 2
      cases' h_one_two with h_one h_two
      · -- charge_sum = 1: cost = 73
        -- RS half-quantum cost formula
        have : gaugeCost s = halfQuantum := by
          apply cost_formula_one_charge
          exact h_one
        rw [this, halfQuantum]
        exact halfQuantum_lt_massGap
      · -- charge_sum = 2: cost = 73
        -- RS half-quantum cost formula
        have : gaugeCost s = halfQuantum := by
          apply cost_formula_two_charges
          exact Nat.lt_succ_iff.mp h_two
        rw [this, halfQuantum]
        exact halfQuantum_lt_massGap

end RecognitionScience.Ledger
