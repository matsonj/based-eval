"""Tests for controllog SDK and builders."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from shared import controllog as cl


class TestControllogGameComplete:
    """Test cases for game_complete event builder."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        cl.init(project_id="test_project", log_dir=Path(self.temp_dir))

    def test_game_complete_event_structure(self):
        """Test that game_complete emits correct event structure."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.game_complete(
                task_id="game:test123",
                project_id="chainlex",
                game_id="test123",
                model_away="claude-3",
                model_home="gpt-4",
                outcome="model_away",
                winner_model="claude-3",
                score_away=15,
                score_home=10,
                margin=5,
                correct_guesses_away=4,
                correct_guesses_home=3,
                total_guesses=8,
                wall_ms=45000,
                run_id="test_run",
            )
            
            # Verify event was written
            assert mock_write.called
            
            # Get the event that was written
            call_args = mock_write.call_args_list[0]  # First call is events.jsonl
            event_data = call_args[0][1]  # Second argument is the data
            
            assert event_data["kind"] == "game_complete"
            assert event_data["project_id"] == "chainlex"
            assert event_data["run_id"] == "test_run"
            
            payload = event_data["payload_json"]
            assert payload["game_id"] == "test123"
            assert payload["model_away"] == "claude-3"
            assert payload["model_home"] == "gpt-4"
            assert payload["outcome"] == "model_away"
            assert payload["winner_model"] == "claude-3"
            assert payload["score_away"] == 15
            assert payload["score_home"] == 10
            assert payload["margin"] == 5
            assert payload["total_guesses"] == 8
            assert payload["wall_ms"] == 45000

    def test_game_complete_with_costs(self):
        """Test game_complete includes cost information."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.game_complete(
                task_id="game:test",
                project_id="chainlex",
                game_id="test",
                model_away="model-a",
                model_home="model-b",
                outcome="tie",
                winner_model=None,
                score_away=10,
                score_home=10,
                margin=0,
                correct_guesses_away=3,
                correct_guesses_home=3,
                total_guesses=6,
                wall_ms=30000,
                cost_money=0.05,
                upstream_cost_money=0.03,
            )
            
            call_args = mock_write.call_args_list[0]
            event_data = call_args[0][1]
            payload = event_data["payload_json"]
            
            assert payload["cost_money"] == 0.05
            assert payload["upstream_cost_money"] == 0.03

    def test_game_complete_with_extra_payload(self):
        """Test game_complete accepts extra payload fields."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.game_complete(
                task_id="game:test",
                project_id="chainlex",
                game_id="test",
                model_away="model-a",
                model_home="model-b",
                outcome="model_home",
                winner_model="model-b",
                score_away=5,
                score_home=20,
                margin=15,
                correct_guesses_away=2,
                correct_guesses_home=5,
                total_guesses=7,
                wall_ms=60000,
                payload={
                    "clue_away": "WATER",
                    "clue_home": "FIRE",
                    "both_hit_assassin": False,
                },
            )
            
            call_args = mock_write.call_args_list[0]
            event_data = call_args[0][1]
            payload = event_data["payload_json"]
            
            assert payload["clue_away"] == "WATER"
            assert payload["clue_home"] == "FIRE"
            assert payload["both_hit_assassin"] is False


class TestControllogModelPromptWithText:
    """Test cases for model_prompt with request_text."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        cl.init(project_id="test_project", log_dir=Path(self.temp_dir))

    def test_model_prompt_includes_request_text(self):
        """Test that model_prompt includes request_text in payload."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.model_prompt(
                task_id="task:test",
                agent_id="agent:test",
                run_id="run:test",
                project_id="test",
                provider="openrouter",
                model="gpt-4",
                prompt_tokens=100,
                request_text="This is the prompt content",
            )
            
            call_args = mock_write.call_args_list[0]
            event_data = call_args[0][1]
            payload = event_data["payload_json"]
            
            assert payload["request_text"] == "This is the prompt content"

    def test_model_prompt_without_request_text(self):
        """Test that model_prompt works without request_text."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.model_prompt(
                task_id="task:test",
                agent_id="agent:test",
                run_id="run:test",
                project_id="test",
                provider="openrouter",
                model="gpt-4",
                prompt_tokens=100,
            )
            
            call_args = mock_write.call_args_list[0]
            event_data = call_args[0][1]
            payload = event_data["payload_json"]
            
            assert "request_text" not in payload


class TestControllogModelCompletionWithText:
    """Test cases for model_completion with response_text."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        cl.init(project_id="test_project", log_dir=Path(self.temp_dir))

    def test_model_completion_includes_response_text(self):
        """Test that model_completion includes response_text in payload."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.model_completion(
                task_id="task:test",
                agent_id="agent:test",
                run_id="run:test",
                project_id="test",
                provider="openrouter",
                model="gpt-4",
                completion_tokens=50,
                wall_ms=1000,
                response_text="This is the model response",
            )
            
            call_args = mock_write.call_args_list[0]
            event_data = call_args[0][1]
            payload = event_data["payload_json"]
            
            assert payload["response_text"] == "This is the model response"

    def test_model_completion_without_response_text(self):
        """Test that model_completion works without response_text."""
        with patch('shared.controllog.sdk._write_jsonl') as mock_write:
            cl.model_completion(
                task_id="task:test",
                agent_id="agent:test",
                run_id="run:test",
                project_id="test",
                provider="openrouter",
                model="gpt-4",
                completion_tokens=50,
                wall_ms=1000,
            )
            
            call_args = mock_write.call_args_list[0]
            event_data = call_args[0][1]
            payload = event_data["payload_json"]
            
            assert "response_text" not in payload


class TestOpenRouterAdapterTextInclusion:
    """Test cases for request/response text inclusion in adapter."""

    def setup_method(self):
        """Setup for each test."""
        from shared.adapters.openrouter_adapter import OpenRouterAdapter
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.adapter = OpenRouterAdapter()

    def test_metadata_includes_request_text(self):
        """Test that metadata includes request_text when include_text=True."""
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        
        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_response):
            response, metadata = self.adapter.call_model_with_metadata(
                "gpt-4", 
                "This is the test prompt",
                include_text=True
            )
        
        assert metadata["request_text"] == "This is the test prompt"
        assert metadata["response_text"] == "Test response"

    def test_metadata_excludes_text_when_disabled(self):
        """Test that metadata excludes text when include_text=False."""
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        
        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_response):
            response, metadata = self.adapter.call_model_with_metadata(
                "gpt-4", 
                "This is the test prompt",
                include_text=False
            )
        
        assert "request_text" not in metadata
        assert "response_text" not in metadata

    def test_metadata_includes_text_by_default(self):
        """Test that text is included by default (include_text defaults to True)."""
        mock_response = {
            "choices": [{"message": {"content": "Default response"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        
        with patch('shared.adapters.openrouter_adapter.chat', return_value=mock_response):
            response, metadata = self.adapter.call_model_with_metadata(
                "gpt-4", 
                "Default test prompt"
            )
        
        assert metadata["request_text"] == "Default test prompt"
        assert metadata["response_text"] == "Default response"

