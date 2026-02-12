"""Receipts endpoint for receipt management."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import hashlib
import json

router = APIRouter()


# In-memory storage for demo (use database in production)
receipts_db: Dict[str, Dict[str, Any]] = {}


class ReceiptRequest(BaseModel):
    """Request to store a receipt."""
    receipt: Dict[str, Any]


class ReceiptResponse(BaseModel):
    """Response for receipt operations."""
    stored: bool
    receipt_id: Optional[str] = None
    digest: Optional[str] = None


class ReceiptListResponse(BaseModel):
    """Response listing receipts."""
    receipts: List[Dict[str, Any]]
    total: int


def compute_receipt_digest(receipt: Dict[str, Any]) -> str:
    """Compute SHA256 digest for a receipt."""
    payload = {k: v for k, v in receipt.items() if k != "digest"}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


@router.post("/receipts", response_model=ReceiptResponse)
async def store_receipt(request: ReceiptRequest):
    """Store a receipt."""
    try:
        receipt = request.receipt
        
        # Compute digest
        digest = compute_receipt_digest(receipt)
        receipt["digest"] = digest
        
        # Store
        receipt_id = receipt.get("event_id", digest)
        receipts_db[receipt_id] = receipt
        
        return ReceiptResponse(
            stored=True,
            receipt_id=receipt_id,
            digest=digest
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/receipts/{receipt_id}", response_model=Dict[str, Any])
async def get_receipt(receipt_id: str):
    """Get a receipt by ID."""
    if receipt_id not in receipts_db:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return receipts_db[receipt_id]


@router.get("/receipts", response_model=ReceiptListResponse)
async def list_receipts():
    """List all receipts."""
    return ReceiptListResponse(
        receipts=list(receipts_db.values()),
        total=len(receipts_db)
    )


@router.delete("/receipts/{receipt_id}")
async def delete_receipt(receipt_id: str):
    """Delete a receipt."""
    if receipt_id not in receipts_db:
        raise HTTPException(status_code=404, detail="Receipt not found")
    del receipts_db[receipt_id]
    return {"deleted": True}
