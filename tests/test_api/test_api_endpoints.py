"""Tests for NSC API endpoints."""
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient
    from api.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestValidateEndpoint:
    """Tests for /api/v1/validate endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_validate_valid_module(self, client):
        """Test validating a valid module."""
        module = {
            "module_id": "test",
            "nodes": [{"id": 1, "kind": "FIELD_REF"}],
            "seq": [1],
            "entrypoints": [1]
        }
        
        response = client.post("/api/v1/validate", json={"module": module})
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert data["module_id"] == "test"

    def test_validate_missing_nodes(self, client):
        """Test validating module with missing nodes."""
        module = {"module_id": "test"}
        
        response = client.post("/api/v1/validate", json={"module": module})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == False
        assert len(data["errors"]) > 0

    def test_validate_invalid_seq_reference(self, client):
        """Test validating module with invalid SEQ references."""
        module = {
            "module_id": "test",
            "nodes": [{"id": 1, "kind": "FIELD_REF"}],
            "seq": [1, 999],  # 999 doesn't exist
            "entrypoints": [1]
        }
        
        response = client.post("/api/v1/validate", json={"module": module})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestExecuteEndpoint:
    """Tests for /api/v1/execute endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_execute_minimal_module(self, client):
        """Test executing a minimal module."""
        module = {
            "module_id": "test",
            "nodes": [{"id": 1, "kind": "FIELD_REF"}],
            "seq": [1],
            "entrypoints": [1]
        }
        
        response = client.post("/api/v1/execute", json={"module": module})
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "execution_time_ms" in data
        assert "module_id" in data

    def test_execute_with_registry(self, client):
        """Test executing with custom registry."""
        module = {
            "module_id": "test",
            "nodes": [{"id": 1, "kind": "FIELD_REF"}],
            "seq": [1],
            "entrypoints": [1]
        }
        registry = {
            "registry_id": "custom",
            "version": "1.0",
            "operators": [
                {"op_id": 1001, "name": "ADD"}
            ]
        }
        
        response = client.post("/api/v1/execute", json={
            "module": module,
            "registry": registry
        })
        assert response.status_code == 200


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestReceiptsEndpoint:
    """Tests for /api/v1/receipts endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_store_receipt(self, client):
        """Test storing a receipt."""
        receipt = {
            "event_id": "evt_001",
            "node_id": 1,
            "op_id": 1001,
            "kernel_id": "kernel.add.v1",
            "kernel_version": "1.0.0"
        }
        
        response = client.post("/api/v1/receipts", json={"receipt": receipt})
        assert response.status_code == 200
        data = response.json()
        assert data["stored"] == True
        assert "receipt_id" in data
        assert "digest" in data

    def test_list_receipts(self, client):
        """Test listing receipts."""
        response = client.get("/api/v1/receipts")
        assert response.status_code == 200
        data = response.json()
        assert "receipts" in data
        assert "total" in data
        assert isinstance(data["receipts"], list)

    def test_get_receipt(self, client):
        """Test getting a specific receipt."""
        # First store one
        receipt = {
            "event_id": "evt_test",
            "node_id": 1,
            "op_id": 1001
        }
        store_response = client.post("/api/v1/receipts", json={"receipt": receipt})
        receipt_id = store_response.json()["receipt_id"]
        
        # Then retrieve it
        response = client.get(f"/api/v1/receipts/{receipt_id}")
        assert response.status_code == 200


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestReplayEndpoint:
    """Tests for /api/v1/replay endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_verify_empty_receipt_chain(self, client):
        """Test verifying empty receipt chain."""
        response = client.post("/api/v1/replay", json={"receipts": []})
        assert response.status_code == 200
        data = response.json()
        assert "receipts_count" in data
        assert "hash_chain_valid" in data
        assert "overall_valid" in data

    def test_verify_receipt_chain(self, client):
        """Test verifying a receipt chain."""
        receipts = [
            {
                "kind": "operator_receipt",
                "event_id": "evt_001",
                "node_id": 1,
                "op_id": 1001
            }
        ]
        module = {
            "module_id": "test",
            "nodes": [{"id": 1, "kind": "APPLY"}]
        }
        
        response = client.post("/api/v1/replay", json={
            "receipts": receipts,
            "module": module
        })
        assert response.status_code == 200
        data = response.json()
        assert data["receipts_count"] == 1
        assert "overall_valid" in data


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
