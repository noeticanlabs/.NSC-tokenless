"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.1.0",
        "service": "nsc-tokenless-api"
    }


@router.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "ready": True,
        "checks": {
            "runtime": True,
            "storage": True,
        }
    }
