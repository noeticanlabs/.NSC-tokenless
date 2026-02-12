"""
NSC Tokenless API - FastAPI Application

Run with: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import json
import time
from contextlib import asynccontextmanager

from api.routes.execute import router as execute_router
from api.routes.validate import router as validate_router
from api.routes.receipts import router as receipts_router
from api.routes.replay import router as replay_router
from api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("NSC Tokenless API starting...")
    yield
    # Shutdown
    print("NSC Tokenless API shutting down...")


app = FastAPI(
    title="NSC Tokenless API",
    description="API for NSC Tokenless v1.1 - Deterministic, auditable execution engine",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(execute_router, prefix="/api/v1", tags=["Execution"])
app.include_router(validate_router, prefix="/api/v1", tags=["Validation"])
app.include_router(receipts_router, prefix="/api/v1", tags=["Receipts"])
app.include_router(replay_router, prefix="/api/v1", tags=["Replay"])


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "NSC Tokenless API",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
