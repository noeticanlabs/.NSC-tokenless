"""Replay endpoint for receipt verification."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter()


class ReplayRequest(BaseModel):
    """Request for replay verification."""
    receipts: List[Dict[str, Any]]
    module: Optional[Dict[str, Any]] = None


class ReplayResponse(BaseModel):
    """Response for replay verification."""
    receipts_count: int
    hash_chain_valid: bool
    overall_valid: bool
    verification_details: List[Dict[str, Any]] = []


@router.post("/replay", response_model=ReplayResponse)
async def verify_replay(request: ReplayRequest):
    """Verify receipt chain through replay."""
    try:
        # Import replay verifier
        from nsc.governance.replay import ReplayVerifier
        
        verifier = ReplayVerifier()
        
        # Verify hash chaining
        chain_valid = verifier.verify_hash_chain(request.receipts)
        
        # Verify each receipt if module provided
        verification_details = []
        if request.module:
            for receipt in request.receipts:
                detail = verifier.verify_receipt(receipt, request.module)
                verification_details.append(detail)
        
        return ReplayResponse(
            receipts_count=len(request.receipts),
            hash_chain_valid=chain_valid,
            verification_details=verification_details,
            overall_valid=chain_valid and all(
                d.get("valid", False) for d in verification_details
            ) if verification_details else chain_valid
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
