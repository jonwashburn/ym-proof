import Mathlib.Topology.Instances.Complex
import YangMillsProof.RecognitionScience

/-- Square-summable complex-valued lattice states. -/
def L2State : Type :=
  { ψ : GaugeLedgerState → ℂ // Summable (fun t => ‖ψ t‖ ^ 2) }

namespace L2State

notation "ℓ²" => L2State

/-- Coerce an `ℓ²` element to the underlying function. -/
instance : CoeFun ℓ² (fun _ => GaugeLedgerState → ℂ) := ⟨Subtype.val⟩

@[simp] lemma summable (ψ : ℓ²) :
    Summable (fun t => ‖ψ t‖ ^ 2) := ψ.property

end L2State
