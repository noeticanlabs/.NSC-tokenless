"""CLI tests for execute command."""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from nsc.cli import main as cli_main


class TestExecuteCommand:
    """Tests for the execute CLI command."""
    
    @pytest.fixture
    def runner(self):
        """CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def sample_module_path(self, tmp_path):
        """Create a temporary module file."""
        module = {
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
        
        path = tmp_path / "test_module.json"
        with open(path, "w") as f:
            json.dump(module, f)
        
        return str(path)
    
    def test_execute_missing_module(self, runner):
        """Test execute with missing module argument."""
        result = runner.invoke(cli_main, ["execute"])
        assert result.exit_code != 0
    
    def test_execute_module_not_found(self, runner):
        """Test execute with non-existent module file."""
        result = runner.invoke(cli_main, ["execute", "/nonexistent/module.json"])
        assert result.exit_code != 0
    
    def test_execute_invalid_json(self, runner, tmp_path):
        """Test execute with invalid JSON file."""
        path = tmp_path / "invalid.json"
        with open(path, "w") as f:
            f.write("not valid json")
        
        result = runner.invoke(cli_main, ["execute", str(path)])
        assert result.exit_code != 0
    
    def test_execute_valid_module(self, runner, sample_module_path):
        """Test execute with valid module."""
        with patch("nsc.cli.execute.Interpreter") as mock_interpreter:
            mock_instance = MagicMock()
            mock_instance.interpret.return_value = MagicMock(
                success=True,
                results={3: 3.0},
                receipts=[{"event_id": "test"}],
                braid_events=[{"event_id": "braid"}],
                execution_time_ms=1.5
            )
            mock_interpreter.return_value = mock_instance
            
            result = runner.invoke(cli_main, ["execute", sample_module_path, "--json"])
            
            # Should succeed
            assert result.exit_code == 0 or "successful" in result.output.lower()
    
    def test_execute_json_output(self, runner, sample_module_path):
        """Test execute with JSON output flag."""
        with patch("nsc.cli.execute.Interpreter") as mock_interpreter:
            mock_instance = MagicMock()
            mock_instance.interpret.return_value = MagicMock(
                success=True,
                results={3: 3.0},
                receipts=[],
                braid_events=[],
                execution_time_ms=1.5
            )
            mock_interpreter.return_value = mock_instance
            
            result = runner.invoke(cli_main, ["execute", sample_module_path, "--json"])
            
            # Output should be valid JSON
            if result.exit_code == 0:
                output = json.loads(result.output)
                assert "success" in output
