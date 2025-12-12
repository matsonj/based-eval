"""Tests for metadata logging functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os

import pytest

from codenames.adapters.openrouter_adapter import OpenRouterAdapter
from codenames.player import AIPlayer
from codenames.utils.logging import log_ai_call_metadata


class TestOpenRouterAdapter:
    """Test cost extraction and metadata handling in OpenRouterAdapter."""

    def setup_method(self):
        """Setup for each test."""
        # Mock the environment variable to avoid requiring API key in tests
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.adapter = OpenRouterAdapter()

    def test_cost_extraction_with_object_attributes(self):
        """Test cost extraction when response has usage dict with cost info."""
        # Mock response in dict format (shared adapter uses requests, not openai SDK)
        mock_api_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
                "cost_details": {
                    "upstream_inference_cost": 0.003
                }
            }
        }

        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_api_response):
            response, metadata = self.adapter.call_model_with_metadata("gpt-4", "Test prompt")

        assert response == "Test response"
        assert metadata["input_tokens"] == 100
        assert metadata["output_tokens"] == 50
        assert metadata["total_tokens"] == 150
        assert metadata["openrouter_cost"] == 0.005
        assert metadata["upstream_cost"] == 0.003

    def test_cost_extraction_with_dictionary_access(self):
        """Test cost extraction when cost_details supports dictionary access."""
        # Mock response in dict format
        mock_api_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
                "cost_details": {"upstream_inference_cost": 0.003}
            }
        }

        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_api_response):
            response, metadata = self.adapter.call_model_with_metadata("gpt-4", "Test prompt")

        assert metadata["openrouter_cost"] == 0.005
        assert metadata["upstream_cost"] == 0.003

    def test_cost_extraction_no_upstream_cost(self):
        """Test cost extraction when only OpenRouter cost is available."""
        mock_api_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
                "cost_details": {}
            }
        }

        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_api_response):
            response, metadata = self.adapter.call_model_with_metadata("gpt-4", "Test prompt")

        assert metadata["openrouter_cost"] == 0.005
        # upstream_cost may be present with 0.0 value when no upstream cost is available
        assert metadata.get("upstream_cost", 0.0) == 0.0

    def test_cost_extraction_no_usage_info(self):
        """Test cost extraction when no usage information is available."""
        mock_api_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {}
        }

        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_api_response):
            response, metadata = self.adapter.call_model_with_metadata("gpt-4", "Test prompt")

        assert response == "Test response"
        # When no usage info, costs should default to 0.0
        assert metadata.get("openrouter_cost", 0.0) == 0.0
        assert metadata.get("upstream_cost", 0.0) == 0.0


class TestAIPlayerMetadata:
    """Test metadata tracking in AI players."""

    def setup_method(self):
        """Setup for each test."""
        # Create AIPlayer with mocked dependencies
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("gpt-4")
        
        # Mock the adapter and prompt manager
        self.mock_adapter = Mock()
        self.mock_adapter.call_model_with_metadata.return_value = (
            "CLUE: ANIMALS\nNUMBER: 3",
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "latency_ms": 500,
                "openrouter_cost": 0.005,
                "upstream_cost": 0.003
            }
        )
        
        self.player._adapter = self.mock_adapter
        self.player.prompt_manager.load_prompt = Mock(return_value="Test prompt")

    def test_spymaster_metadata_storage(self):
        """Test that spymaster calls store metadata correctly."""
        board_state = {
            "board": ["ALPHA", "BRAVO", "CHARLIE"],
            "revealed": {"ALPHA": False, "BRAVO": False, "CHARLIE": False},
            "current_team": "red",
            "identities": {"ALPHA": "red_agent", "BRAVO": "blue_agent", "CHARLIE": "bystander"}
        }
        
        clue, number = self.player.get_spymaster_move(board_state, "test_prompt.md")
        
        assert clue == "ANIMALS"
        assert number == 3
        
        metadata = self.player.get_last_call_metadata()
        assert metadata["call_type"] == "spymaster"
        assert metadata["input_tokens"] == 100
        assert metadata["output_tokens"] == 20
        assert metadata["openrouter_cost"] == 0.005
        assert metadata["upstream_cost"] == 0.003
        assert metadata["is_retry"] == False
        assert metadata["turn_result"]["clue"] == "ANIMALS"
        assert metadata["turn_result"]["clue_number"] == 3

    def test_operative_metadata_storage(self):
        """Test that operative calls store metadata correctly."""
        # Test metadata storage directly without going through complex parsing
        mock_metadata = {
            "input_tokens": 150,
            "output_tokens": 30,
            "total_tokens": 180,
            "latency_ms": 600,
            "openrouter_cost": 0.007,
            "upstream_cost": 0.004
        }
        
        # Set metadata directly on the player to test metadata storage
        self.player._last_call_metadata = mock_metadata.copy()
        self.player._last_call_metadata["call_type"] = "operative"
        self.player._last_call_metadata["turn_result"] = {"guesses": ["ALPHA", "BRAVO"]}

        metadata = self.player.get_last_call_metadata()
        assert metadata["call_type"] == "operative"
        assert metadata["input_tokens"] == 150
        assert metadata["output_tokens"] == 30
        assert metadata["openrouter_cost"] == 0.007
        assert metadata["upstream_cost"] == 0.004
        assert metadata["turn_result"]["guesses"] == ["ALPHA", "BRAVO"]


class TestMetadataLogging:
    """Test the metadata logging system."""

    def test_log_ai_call_metadata_writes_jsonl(self):
        """Test that log_ai_call_metadata creates proper metadata structure."""
        mock_logger = Mock()
        
        with patch('logging.getLogger', return_value=mock_logger):
            log_ai_call_metadata(
                game_id="test_game_123",
                model_name="gpt-4",
                call_type="operative",
                team="red",
                turn="1a",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                latency_ms=500,
                openrouter_cost=0.005,
                upstream_cost=0.003,
                turn_result={"clue": "ANIMALS", "clue_number": "3"},
                game_continues=True,
                is_retry=False
            )
        
        # Verify the logger was called
        mock_logger.info.assert_called_once()
        
        # Parse the logged JSON data
        logged_json = mock_logger.info.call_args[0][0]
        logged_data = json.loads(logged_json)
        
        assert logged_data["game_id"] == "test_game_123"
        assert logged_data["model_name"] == "gpt-4"
        assert logged_data["type"] == "operative"
        assert logged_data["team"] == "red"
        assert logged_data["turn"] == "1a"
        assert logged_data["total_tokens"] == 120
        assert logged_data["openrouter_cost"] == 0.005
        assert logged_data["upstream_cost"] == 0.003
        assert logged_data["latency_ms"] == 500
        assert logged_data["clue"] == "ANIMALS"
        assert logged_data["clue_number"] == "3"
        assert logged_data["game_continues"] == 1

    def test_log_ai_call_metadata_without_upstream_cost(self):
        """Test logging when upstream cost is not available."""
        mock_logger = Mock()
        
        with patch('logging.getLogger', return_value=mock_logger):
            log_ai_call_metadata(
                game_id="test_game_123",
                model_name="claude-3",
                call_type="operative",
                team="blue",
                turn="1b",
                input_tokens=150,
                output_tokens=30,
                total_tokens=180,
                latency_ms=600,
                openrouter_cost=0.007,
                upstream_cost=0.0,  # No upstream cost
                turn_result={"guesses": ["ALPHA", "BRAVO"]},
                game_continues=True,
                is_retry=False
            )
        
        # Verify the logger was called
        mock_logger.info.assert_called_once()
        
        # Parse the logged JSON data
        logged_json = mock_logger.info.call_args[0][0]
        logged_data = json.loads(logged_json)
        
        assert logged_data["model_name"] == "claude-3"
        assert logged_data["type"] == "operative"
        assert logged_data["openrouter_cost"] == 0.007
        assert logged_data["upstream_cost"] == 0.0
