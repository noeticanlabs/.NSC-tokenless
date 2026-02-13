"""Phase 2 benchmark: compare baseline vs guided ranking.

This is a lightweight benchmark that:
- Synthesizes candidate chains with beam search.
- Reranks candidates with the Phase 2 predictor.
- Compares time and predicted success of top-ranked chains.

Run:
    /workspaces/noetica/.venv/bin/python scripts/phase2_benchmark.py
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Dict, List, Tuple

from nsc.npe_v2_0 import (
    BeamSearchSynthesizer,
    DebtFunctional,
    ExecutionTracer,
    GateConfig,
    OperatorEmbeddingLearner,
    OperatorReceipt,
    OperatorRegistry,
    KernelBinding,
    Operator,
    OpSig,
    ReceiptLedger,
    SCALAR,
    SuccessPredictorModel,
    FIELD,
)


def build_registry() -> Tuple[OperatorRegistry, KernelBinding]:
    ops = {
        10: Operator(10, "source_plus", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                     "nsc.apply.source_plus", ["kernel_id", "kernel_version"], "res.op.source_plus"),
        11: Operator(11, "twist", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                     "nsc.apply.curvature_twist", ["kernel_id", "kernel_version"], "res.op.twist"),
        12: Operator(12, "delta", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                     "nsc.apply.delta_transform", ["kernel_id", "kernel_version"], "res.op.delta"),
        13: Operator(13, "regulate", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                     "nsc.apply.regulate_circle", ["kernel_id", "kernel_version"], "res.op.regulate"),
        14: Operator(14, "sink_minus", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                     "nsc.apply.sink_minus", ["kernel_id", "kernel_version"], "res.op.sink_minus"),
        20: Operator(20, "return_to_baseline", OpSig([FIELD(SCALAR), FIELD(SCALAR)], FIELD(SCALAR)),
                     "nsc.apply.return_to_baseline", ["kernel_id", "kernel_version", "params"], "res.op.return_baseline"),
    }

    registry = OperatorRegistry("noetican.nsc.registry.v1_1", "1.1.0", ops)
    binding = KernelBinding(
        registry_id=registry.registry_id,
        binding_id="binding.demo.v1",
        bindings={
            10: ("kern.source_plus.v1", "1.0.0"),
            11: ("kern.twist.v1", "1.0.0"),
            12: ("kern.delta.v1", "1.0.0"),
            13: ("kern.regulate.v1", "1.0.0"),
            14: ("kern.sink_minus.v1", "1.0.0"),
            20: ("kern.return_baseline.linear_blend", "1.0.0"),
        },
    )

    return registry, binding


def make_ledger(chain_ops: List[int], success: bool) -> ReceiptLedger:
    ledger = ReceiptLedger()
    c_before = 0.0
    debt_step = 0.02 if success else 0.12

    for i, op_id in enumerate(chain_ops, start=1):
        c_after = c_before + debt_step
        receipt = OperatorReceipt(
            event_id=f"evt_{i}",
            node_id=i,
            op_id=op_id,
            digest_in=f"sha256:{'0'*64}",
            digest_out="sha256:" + "1" * 64,
            kernel_id="kern.demo",
            kernel_version="1.0.0",
            C_before=c_before,
            C_after=c_after,
            r_phys=0.02 if success else 0.15,
            r_cons=0.02 if success else 0.12,
            r_num=0.01 if success else 0.10,
            registry_id="noetican.nsc.registry.v1_1",
            registry_version="1.1.0",
        )
        prev_hash = ledger.hash_chain[-1] if ledger.hash_chain else None
        ledger.add_receipt(receipt, prev_hash)
        c_before = c_after

    return ledger


def train_tracer() -> Tuple[ExecutionTracer, Dict[int, List[float]]]:
    tracer = ExecutionTracer()

    good_chains = [[10, 11, 12], [10, 13, 12], [10, 11, 14]]
    bad_chains = [[14, 20, 11], [20, 14, 13]]

    for chain in good_chains:
        tracer.ingest_receipts(make_ledger(chain, success=True))
    for chain in bad_chains:
        tracer.ingest_receipts(make_ledger(chain, success=False))

    learner = OperatorEmbeddingLearner(embedding_dim=64)
    embeddings = learner.train(tracer)

    return tracer, {op_id: emb.embedding for op_id, emb in embeddings.items()}


def train_predictor(embeddings: Dict[int, List[float]]) -> SuccessPredictorModel:
    predictor = SuccessPredictorModel(embedding_dim=64)

    chains = [
        [embeddings[10], embeddings[11], embeddings[12]],
        [embeddings[10], embeddings[13], embeddings[12]],
        [embeddings[10], embeddings[11], embeddings[14]],
        [embeddings[14], embeddings[20], embeddings[11]],
        [embeddings[20], embeddings[14], embeddings[13]],
    ]
    labels = [1, 1, 1, 0, 0]
    predictor.train(chains, labels, epochs=15)

    return predictor


def score_chain(chain_nodes: List[Dict[str, object]],
                embeddings: Dict[int, List[float]],
                predictor: SuccessPredictorModel) -> float:
    op_ids = [node.get("op_id", 0) for node in chain_nodes]
    chain_embeddings = [embeddings[op] for op in op_ids if op in embeddings]
    return predictor.predict(chain_embeddings)


def benchmark(iterations: int = 25) -> None:
    registry, binding = build_registry()
    synthesizer = BeamSearchSynthesizer(
        registry=registry,
        binding=binding,
        debt_functional=DebtFunctional(),
        gate_config=GateConfig(),
        beam_width=5,
        max_depth=5,
    )

    _, embeddings = train_tracer()
    predictor = train_predictor(embeddings)

    input_type = FIELD(SCALAR).to_json()
    target_type = FIELD(SCALAR).to_json()

    baseline_times = []
    guided_times = []
    baseline_scores = []
    guided_scores = []
    fallback_used = False

    for _ in range(iterations):
        start = time.perf_counter()
        chains = synthesizer.synthesize(input_type, target_type, context={})
        if not chains:
            fallback_used = True
            fallback_ops = list(embeddings.keys())[:3]
            chains = [([{"op_id": op_id}], 0.0, None) for op_id in fallback_ops]
        baseline_times.append(time.perf_counter() - start)

        # Baseline: use raw cost ordering
        if chains:
            top_chain = chains[0][0]
            baseline_scores.append(score_chain(top_chain, embeddings, predictor))

        start = time.perf_counter()
        reranked = []
        for chain_nodes, chain_cost, _ in chains:
            prob = score_chain(chain_nodes, embeddings, predictor)
            reranked.append((chain_nodes, chain_cost + (1.0 - prob)))
        reranked.sort(key=lambda x: x[1])
        guided_times.append(time.perf_counter() - start)

        if reranked:
            guided_scores.append(score_chain(reranked[0][0], embeddings, predictor))

    print("=" * 70)
    print("Phase 2 Benchmark: Baseline vs Guided Rerank")
    print("=" * 70)
    print(f"Iterations: {iterations}")
    print(f"Baseline synth avg time: {mean(baseline_times):.6f}s")
    print(f"Guided rerank avg time:  {mean(guided_times):.6f}s")
    if fallback_used:
        print("Note: No chains returned by synthesizer; using fallback operators for scoring.")
    if baseline_scores and guided_scores:
        print(f"Baseline top predicted success: {mean(baseline_scores):.3f}")
        print(f"Guided top predicted success:   {mean(guided_scores):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark()
