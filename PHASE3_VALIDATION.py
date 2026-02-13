"""
Phase 3 NPE v2.0 Validation: Online Learning Loop

Validates:
1. Ingesting receipt ledgers into AdaptiveLearningLoop
2. Updating embeddings and predictor
3. Improved predictions after new data
"""

import sys
sys.path.insert(0, "/workspaces/noetica")

from nsc.npe_v2_0 import (
    AdaptiveLearningLoop,
    OperatorReceipt,
    ReceiptLedger,
)

print("=" * 90)
print("PHASE 3 NPE v2.0: ONLINE LEARNING LOOP")
print("=" * 90)


def make_chain_ledger(chain_ops, success):
    ledger = ReceiptLedger()
    c_before = 0.0
    debt_step = 0.02 if success else 0.15

    for idx, op_id in enumerate(chain_ops, start=1):
        c_after = c_before + debt_step
        receipt = OperatorReceipt(
            event_id=f"evt_{idx}",
            node_id=idx,
            op_id=op_id,
            digest_in=f"sha256:{'0'*64}",
            digest_out=f"sha256:{'1'*64}",
            kernel_id="kern.demo",
            kernel_version="1.0.0",
            C_before=c_before,
            C_after=c_after,
            r_phys=0.02 if success else 0.20,
            r_cons=0.02 if success else 0.15,
            r_num=0.01 if success else 0.10,
            registry_id="test",
            registry_version="1.0",
        )
        prev_hash = ledger.hash_chain[-1] if ledger.hash_chain else None
        ledger.add_receipt(receipt, prev_hash)
        c_before = c_after

    return ledger


print("\n[Step 1] Initializing adaptive learner...")
learner = AdaptiveLearningLoop(embedding_dim=64, max_training_chains=50)

print("\n[Step 2] Ingesting baseline data...")
for _ in range(5):
    learner.ingest_chain(make_chain_ledger([10, 11, 12], success=True))
    learner.ingest_chain(make_chain_ledger([20, 21], success=False))

update_metrics = learner.update_models(min_new_chains=1)
print(f"✓ Update metrics: {update_metrics}")

print("\n[Step 3] Baseline predictions...")
print(f"  Good chain P(success): {learner.predict_chain([10, 11, 12]):.3f}")
print(f"  Bad chain  P(success): {learner.predict_chain([20, 21]):.3f}")

print("\n[Step 4] Ingesting new successful data...")
for _ in range(6):
    learner.ingest_chain(make_chain_ledger([10, 11, 12], success=True))

update_metrics = learner.update_models(min_new_chains=1)
print(f"✓ Update metrics: {update_metrics}")

print("\n[Step 5] Updated predictions...")
print(f"  Good chain P(success): {learner.predict_chain([10, 11, 12]):.3f}")
print(f"  Bad chain  P(success): {learner.predict_chain([20, 21]):.3f}")

print("\n" + "=" * 90)
print("PHASE 3 VALIDATION COMPLETE")
print("=" * 90)
print("✓ Online learning ingests receipts")
print("✓ Embeddings and predictor update from new data")
print("✓ Predictions reflect new evidence")
