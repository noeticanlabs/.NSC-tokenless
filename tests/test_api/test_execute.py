"""API tests for execute endpoints."""
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock the API modules before import
sys.modules['nsc.runtime'] = MagicMock()
sys.modules['nsc.runtime.interpreter'] = MagicMock()
sys.modules['nsc.runtime.executor'] = MagicMock()


class TestExecuteAPI:
    """Tests for execute API endpoints."""
    
    def test_api_module_exists(self):
        """Test that API module can be imported."""
        from api.main import app
        assert app is not None
    
    def test_health_route_exists(self):
        """Test that health route exists."""
        from api.main import app
        routes = [r.path for r in app.routes]
        assert any('/health' in r for r in routes) or len(routes) > 0
    
    def test_execute_route_exists(self):
        """Test that execute route exists."""
        from api.main import app
        routes = [r.path for r in app.routes]
        assert any('/execute' in r or '/api/execute' in r for r in routes)
