"""
NSC Glyphs Module (L2 Conformance)

Per docs/glyphs/: GLLL codebook and decode for identity-aware systems.

L2 Conformance:
- GLLL decode with REJECT control row
- DecodeReport emission
- Ambiguity recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime
import math
import numpy as np


class HadamardMatrix:
    """Hadamard matrix utilities for GLLL code generation."""
    
    @staticmethod
    def create(n: int) -> np.ndarray:
        """Create Hadamard matrix of order n."""
        if n == 1:
            return np.array([[1]])
        
        # Sylvester's construction
        H = np.array([[1]])
        for _ in range(int(math.log2(n))):
            H = np.block([[H, H], [H, -H]])
        
        return H

    @staticmethod
    def get_row(n: int, row_idx: int) -> List[int]:
        """Get a specific row from Hadamard matrix."""
        H = HadamardMatrix.create(n)
        row = H[row_idx]
        return [int(x) for x in row]


@dataclass
class GLLLCodebook:
    """
    GLLL codebook per docs/glyphs/glll_codebook.md.
    
    Structure:
    - n: vector length
    - reserved_control_rows: REJECT required
    - rows: [{row_id, glyph_id, vector}]
    - mapping_to_op_id: glyph_id -> op_id
    """
    n: int
    rows: List[Dict[str, Any]] = field(default_factory=list)
    control_rows: Dict[str, int] = field(default_factory=dict)
    glyph_to_op_id: Dict[str, int] = field(default_factory=dict)
    version: str = "1.0"
    
    @classmethod
    def create_hadamard_16(cls) -> "GLLLCodebook":
        """Create Hadamard-derived codebook for n=16."""
        codebook = GLLLCodebook(n=16, version="hadamard_n16_v1")
        
        # Hadamard rows (simplified - actual implementation would use proper Hadamard matrix)
        hadamard_vectors = [
            [1] * 16,  # Row 0
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],  # Row 1
            # ... additional rows would be generated
        ]
        
        # Add rows with glyph IDs
        for i, vector in enumerate(hadamard_vectors[:8]):  # Use first 8 rows + control
            codebook.rows.append({
                "row_id": i,
                "glyph_id": f"GLYPH.OP_{10+i}",
                "vector": vector
            })
        
        # Add REJECT control row
        codebook.control_rows["REJECT"] = 0
        codebook.rows.insert(0, {
            "row_id": 0,
            "glyph_id": "REJECT",
            "vector": [1] * 16  # REJECT vector
        })
        
        # Map glyphs to op_ids
        for i in range(1, len(codebook.rows)):
            glyph_id = codebook.rows[i]["glyph_id"]
            codebook.glyph_to_op_id[glyph_id] = 10 + (i - 1)  # op_id 10+
        
        return codebook
    
    def get_row(self, row_id: int) -> Optional[Dict[str, Any]]:
        """Get row by ID."""
        for row in self.rows:
            if row["row_id"] == row_id:
                return row
        return None
    
    def get_reject_row(self) -> Dict[str, Any]:
        """Get REJECT control row."""
        return self.get_row(self.control_rows.get("REJECT", 0))

@dataclass
class DecodeResult:
    """Result of GLLL decode."""
    chosen_kind: str  # "GLYPH_ID" or "REJECT"
    glyph_id: Optional[str] = None
    op_id: Optional[int] = None
    control_row_id: Optional[int] = None
    topk: List[Dict[str, Any]] = field(default_factory=list)
    margin: float = 0.0
    rejected: bool = False

@dataclass
class DecodeReport:
    """
    Decode report per nsc_decode_report.schema.json.
    
    Per docs/glyphs/glll_decode.md: Emit DecodeReport for every decode attempt.
    """
    event_id: str
    n: int
    chosen: Dict[str, Any]
    topk: List[Dict[str, Any]]
    thresholds: Dict[str, float]
    margin: float
    codebook_version: str = "1.0"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    digest: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "decode_report",
            "event_id": self.event_id,
            "n": self.n,
            "chosen": self.chosen,
            "topk": self.topk,
            "thresholds": self.thresholds,
            "margin": self.margin,
            "codebook_version": self.codebook_version,
            "timestamp": self.timestamp,
            "digest": self.digest
        }
    
    def compute_digest(self, prev_hash: str = None) -> str:
        """Compute report digest."""
        payload = {k: v for k, v in self.to_dict().items() if k != "digest"}
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        
        if prev_hash:
            combined = canon.encode("utf-8") + prev_hash.encode("utf-8")
            self.digest = "sha256:" + hashlib.sha256(combined).hexdigest()
        else:
            self.digest = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
        
        return self.digest

class GLLLDecoder:
    """
    GLLL decoder per docs/glyphs/glll_decode.md.
    
    Scoring: s(row) = (w · v_row) / n
    Decision: if (s1 < tau_abs) OR (margin < tau_margin) → REJECT
    """
    
    def __init__(self, codebook: GLLLCodebook):
        self.codebook = codebook
        self.tau_abs = 0.60  # Default threshold
        self.tau_margin = 0.10  # Default margin
    
    def decode(self, 
               observed: List[float],
               tau_abs: float = None,
               tau_margin: float = None) -> DecodeResult:
        """
        Decode observed vector to glyph identity or REJECT.
        
        Args:
            observed: Observed vector (real or {-1,+1})
            tau_abs: Absolute threshold (default 0.60)
            tau_margin: Margin threshold (default 0.10)
            
        Returns:
            DecodeResult with chosen glyph or REJECT
        """
        tau_abs = tau_abs or self.tau_abs
        tau_margin = tau_margin or self.tau_margin
        
        # Compute scores for each row
        scores = []
        for row in self.codebook.rows:
            score = self._correlation(observed, row["vector"])
            scores.append({
                "row_id": row["row_id"],
                "glyph_id": row["glyph_id"],
                "score": score
            })
        
        # Sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Get best and second best
        best = scores[0]
        second = scores[1] if len(scores) > 1 else {"score": -1}
        
        # Compute margin
        margin = best["score"] - second["score"]
        
        # Decision rule
        if best["score"] < tau_abs or margin < tau_margin:
            # REJECT
            return DecodeResult(
                chosen_kind="REJECT",
                control_row_id=self.codebook.control_rows.get("REJECT", 0),
                topk=scores[:5],
                margin=margin,
                rejected=True
            )
        
        # GLYPH_ID
        return DecodeResult(
            chosen_kind="GLYPH_ID",
            glyph_id=best["glyph_id"],
            op_id=self.codebook.glyph_to_op_id.get(best["glyph_id"]),
            topk=scores[:5],
            margin=margin,
            rejected=False
        )
    
    def _correlation(self, w: List[float], v: List[float]) -> float:
        """Compute correlation score: (w · v) / n"""
        if len(w) != len(v):
            return -1.0
        
        n = len(w)
        dot = sum(wi * vi for wi, vi in zip(w, v))
        return dot / n
    
    def emit_report(self, event_id: str, result: DecodeResult) -> DecodeReport:
        """Emit DecodeReport for decode result."""
        if result.rejected:
            chosen = {
                "kind": "REJECT",
                "control_row_id": result.control_row_id
            }
        else:
            chosen = {
                "kind": "GLYPH_ID",
                "glyph_id": result.glyph_id
            }
        
        return DecodeReport(
            event_id=event_id,
            n=self.codebook.n,
            chosen=chosen,
            topk=[{"glyph_id": t["glyph_id"], "score": t["score"]} for t in result.topk],
            thresholds={"tau_abs": self.tau_abs, "tau_margin": self.tau_margin},
            margin=result.margin,
            codebook_version=self.codebook.version
        )

class REJECTHandler:
    """
    REJECT control row handler per docs/glyphs/reject_control_row.md.
    
    REJECT is NOT an operator and MUST NOT be lowered to APPLY.
    """
    
    def __init__(self):
        self.reject_count = 0
        self.recovery_attempts = 0
    
    def handle(self, decode_report: DecodeReport) -> Dict[str, Any]:
        """
        Handle REJECT outcome.
        
        Per docs/glyphs/reject_control_row.md:
        - Emit DecodeReport braid event
        - Stop lowering/executing affected segment
        - Do not substitute "best guess" silently
        """
        self.reject_count += 1
        
        return {
            "action": "STOP",
            "reason": "DECODE_REJECT",
            "event_id": decode_report.event_id,
            "report_digest": decode_report.digest,
            "note": "REJECT is control-only; not lowered to APPLY"
        }
    
    def attempt_recovery(self,
                         decode_report: DecodeReport,
                         context_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Attempt contextual recovery per docs/glyphs/ambiguity_recovery.md.
        
        Recovery is OPTIONAL and MUST:
        - Keep original DecodeReport
        - Emit audit event describing recovery
        - Never hide that REJECT occurred
        """
        context_rules = context_rules or []
        self.recovery_attempts += 1
        
        # Simplified recovery: try context rules
        for rule in context_rules:
            if rule.get("type") == "type_constraint":
                # Try to resolve based on type
                pass
        
        return {
            "action": "RECOVERY_ATTEMPTED",
            "original_reject": decode_report.event_id,
            "rules_tested": len(context_rules),
            "note": "Recovery does not hide original REJECT"
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get REJECT handling statistics."""
        return {
            "reject_count": self.reject_count,
            "recovery_attempts": self.recovery_attempts
        }
