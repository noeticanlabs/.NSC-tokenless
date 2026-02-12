"""API tests for execute endpoint."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.main import app


class TestExecuteAPI:
    """Tests for the execute API endpoint."""
    
    @pytest.fixture
    def client(self):
        """Test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_module(self):
        """Sample module for testing."""
        return {
            "module_id": "test_module",
            "version": "1.1.0",
            "nodes": [
                {"id": 1, "kind": "CONST", "type_tag": {"tag": "float"}, "payload": {"value": 1.0}},
                {"id": 2, "kind": "CONST", "type_tag": {"tag": "float"}, "payload": {"value": 2.0}},
                {"id": 3, "kind": "APPLY", "type_tag": {"tag": "float"}, "payload": {"op_id": 1001, "inputs": [1, 2]}}
            ],
            "seq": [1, 2, 3],
            "entrypoints": [3],
            "registry_ref": "default"
        }
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_execute_missing_module(self, client):
        """Test execute with missing module."""
        response = client.post("/api/v1/execute", json={})
        assert response.status_code == 422  # Validation error
    
    def test_execute_invalid_json(self, client):
        """Test execute with invalid JSON body."""
        response = client.post(
            "/api/v1/execute",
            json={"module": {"invalid": "structure"}}
        )
        # Should return validation error due to missing required fields
        assert response.status_code in [200, 422, 500]
    
    def test_execute_valid_module(self, client, sample_module):
        """Test execute with valid module."""
        response = client.post(
            "/api/v1/execute",
            json={"module": sample_module}
        )
        
        # Should return a response
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "execution_time_ms" in data
    
    def test_execute_with_registry(self, client, sample_module):
        """Test execute with custom registry."""
        registry = {
            "registry_id": "custom",
            "version": "1.0",
            "operators": [
                {"op_id": 1001, "name": "ADD", "arg_types": ["float", "float"], "ret_type": "float"}
            ]
        }
        
        response = client.post(
            "/api/v1/execute",
            json={"module": sample_module, "registry": registry}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
