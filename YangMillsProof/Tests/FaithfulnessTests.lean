import YangMillsProof.Stage1_GaugeEmbedding.GaugeToLedger

namespace YangMillsProof.Tests

open Stage1_GaugeEmbedding

/-- Property test: random connections map to distinct ledgers -/
def test_injectivity (N : ℕ) (scale : LatticeScale) : Prop :=
  ∀ A A' : Connection N, 
    wilsonLoop A ≠ wilsonLoop A' → 
    ledgerOfConnection scale A ≠ ledgerOfConnection scale A'

end YangMillsProof.Tests
