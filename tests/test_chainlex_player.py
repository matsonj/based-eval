"""Tests for ChainLex-1 AI player functionality."""

import random
from unittest.mock import Mock, patch, MagicMock

import pytest

from chainlex.player import AIPlayer


class TestChainLexAIPlayer:
    """Test cases for ChainLex AIPlayer."""

    def setup_method(self):
        """Setup for each test."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("test-model")

    def test_player_initialization(self):
        """Test player initializes correctly."""
        assert self.player.model_name == "test-model"
        assert self.player._last_call_metadata is None

    def test_adapter_lazy_initialization(self):
        """Test adapter is lazily initialized."""
        assert self.player._adapter is None
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            _ = self.player.adapter  # Access adapter
            assert self.player._adapter is not None


class TestChainLexClueGiverParsing:
    """Test cases for clue giver response parsing."""

    def setup_method(self):
        """Setup for each test."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("test-model")

    def test_parse_standard_format(self):
        """Test parsing standard CLUE/NUMBER format."""
        response = "CLUE: WATER\nNUMBER: 3"
        clue, number = self.player._parse_clue_giver_response(response)
        assert clue == "WATER"
        assert number == 3

    def test_parse_markdown_format(self):
        """Test parsing markdown-formatted response."""
        response = "**CLUE:** FIRE\n**NUMBER:** 2"
        clue, number = self.player._parse_clue_giver_response(response)
        assert clue == "FIRE"
        assert number == 2

    def test_parse_colon_format(self):
        """Test parsing 'clue: number' format."""
        response = "OCEAN: 4"
        clue, number = self.player._parse_clue_giver_response(response)
        assert clue == "OCEAN"
        assert number == 4

    def test_parse_number_bounds(self):
        """Test that number is bounded correctly."""
        # Test minimum bound
        response = "CLUE: TEST\nNUMBER: 0"
        clue, number = self.player._parse_clue_giver_response(response)
        assert number == 1  # Should be at least 1

        # Test maximum bound
        response = "CLUE: TEST\nNUMBER: 99"
        clue, number = self.player._parse_clue_giver_response(response)
        assert number == 8  # Should be at most 8

    def test_parse_strips_quotes(self):
        """Test that quotes are stripped from clue."""
        response = "CLUE: \"QUOTED\"\nNUMBER: 2"
        clue, number = self.player._parse_clue_giver_response(response)
        assert clue == "QUOTED"


class TestChainLexGuesserParsing:
    """Test cases for guesser response parsing."""

    def setup_method(self):
        """Setup for each test."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("test-model")
        self.board_state = {
            "board": ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", 
                     "FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIET",
                     "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA"],
            "revealed": {word: False for word in ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", 
                                                   "FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIET",
                                                   "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA"]},
        }

    def test_parse_simple_list(self):
        """Test parsing simple word list."""
        response = "ALPHA\nBRAVO\nCHARLIE"
        guesses = self.player._parse_guesser_response(response, self.board_state, 3)
        assert guesses == ["ALPHA", "BRAVO", "CHARLIE"]

    def test_parse_comma_separated(self):
        """Test parsing comma-separated words."""
        response = "ALPHA, BRAVO, CHARLIE"
        guesses = self.player._parse_guesser_response(response, self.board_state, 3)
        assert guesses == ["ALPHA", "BRAVO", "CHARLIE"]

    def test_parse_respects_max_number(self):
        """Test that guesses are limited to number + 1."""
        response = "ALPHA\nBRAVO\nCHARLIE\nDELTA\nECHO\nFOXTROT"
        guesses = self.player._parse_guesser_response(response, self.board_state, 2)
        assert len(guesses) <= 3  # number + 1

    def test_parse_ignores_invalid_words(self):
        """Test that words not on board are ignored."""
        response = "ALPHA\nINVALID_WORD\nBRAVO"
        guesses = self.player._parse_guesser_response(response, self.board_state, 3)
        assert "INVALID_WORD" not in guesses
        assert "ALPHA" in guesses
        assert "BRAVO" in guesses

    def test_parse_skips_revealed_words(self):
        """Test that revealed words are not included."""
        self.board_state["revealed"]["ALPHA"] = True
        response = "ALPHA\nBRAVO\nCHARLIE"
        guesses = self.player._parse_guesser_response(response, self.board_state, 3)
        assert "ALPHA" not in guesses

    def test_parse_no_duplicates(self):
        """Test that duplicate guesses are removed."""
        response = "ALPHA\nBRAVO\nALPHA\nCHARLIE"
        guesses = self.player._parse_guesser_response(response, self.board_state, 4)
        assert guesses.count("ALPHA") == 1

    def test_parse_fallback_on_empty(self):
        """Test fallback to first available word when no valid guesses."""
        response = "INVALID1\nINVALID2"
        guesses = self.player._parse_guesser_response(response, self.board_state, 2)
        assert len(guesses) == 1
        assert guesses[0] in self.board_state["board"]


class TestChainLexRefereeParsing:
    """Test cases for referee response parsing."""

    def setup_method(self):
        """Setup for each test."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("test-model")

    def test_parse_valid_response(self):
        """Test parsing VALID response."""
        response = "VALID: The clue follows all rules"
        is_valid, reasoning = self.player._parse_referee_response(response)
        assert is_valid is True
        assert "follows all rules" in reasoning

    def test_parse_invalid_response(self):
        """Test parsing INVALID response."""
        response = "INVALID: The clue violates rule 1"
        is_valid, reasoning = self.player._parse_referee_response(response)
        assert is_valid is False
        assert "violates rule 1" in reasoning

    def test_parse_invalid_with_violation(self):
        """Test parsing INVALID with Violation line."""
        response = "INVALID\nViolation: Contains a board word"
        is_valid, reasoning = self.player._parse_referee_response(response)
        assert is_valid is False
        assert "Contains a board word" in reasoning


class TestChainLexMetadataStorage:
    """Test cases for metadata storage in AI calls."""

    def setup_method(self):
        """Setup for each test."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("test-model")
        
        self.mock_adapter = Mock()
        self.mock_adapter.call_model_with_metadata.return_value = (
            "CLUE: TEST\nNUMBER: 2",
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "latency_ms": 500,
                "openrouter_cost": 0.005,
                "upstream_cost": 0.003,
                "request_text": "Test prompt",
                "response_text": "CLUE: TEST\nNUMBER: 2",
            }
        )
        self.player._adapter = self.mock_adapter
        self.player.prompt_manager.load_prompt = Mock(return_value="Test prompt")

    def test_clue_giver_stores_metadata(self):
        """Test that clue giver calls store metadata."""
        board_state = {
            "board": ["WORD1", "WORD2"],
            "revealed": {"WORD1": False, "WORD2": False},
            "identities": {"WORD1": "friendly", "WORD2": "bystander"},
            "score": 0,
            "correct_guesses": 0,
        }
        
        clue, number = self.player.get_clue_giver_move(board_state, "test_prompt.md")
        
        metadata = self.player.get_last_call_metadata()
        assert metadata is not None
        assert metadata["call_type"] == "clue_giver"
        assert metadata["input_tokens"] == 100
        assert metadata["openrouter_cost"] == 0.005
        assert "request_text" in metadata
        assert "response_text" in metadata

    def test_guesser_stores_metadata(self):
        """Test that guesser calls store metadata."""
        self.mock_adapter.call_model_with_metadata.return_value = (
            "WORD1\nWORD2",
            {
                "input_tokens": 80,
                "output_tokens": 10,
                "total_tokens": 90,
                "latency_ms": 300,
                "openrouter_cost": 0.003,
                "upstream_cost": 0.002,
                "request_text": "Guesser prompt",
                "response_text": "WORD1\nWORD2",
            }
        )
        
        # Create a proper 16-element board (4x4 grid)
        words = [f"WORD{i}" for i in range(1, 17)]
        board_state = {
            "board": words,
            "revealed": {word: False for word in words},
            "identities": {},
            "score": 0,
            "correct_guesses": 0,
        }
        
        guesses = self.player.get_guesser_moves(board_state, "CLUE", 2, "test_prompt.md")
        
        metadata = self.player.get_last_call_metadata()
        assert metadata is not None
        assert metadata["call_type"] == "guesser"
        assert "guesses" in metadata["turn_result"]


class TestChainLexBoardFormatting:
    """Test cases for board formatting."""

    def setup_method(self):
        """Setup for each test."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            self.player = AIPlayer("test-model")

    def test_format_board_for_guesser(self):
        """Test board formatting for guesser display."""
        board_state = {
            "board": ["A", "B", "C", "D", "E", "F", "G", "H",
                     "I", "J", "K", "L", "M", "N", "O", "P"],
            "revealed": {letter: False for letter in "ABCDEFGHIJKLMNOP"},
        }
        
        formatted = self.player._format_board_for_guesser(board_state)
        
        # Should have 4 lines (4x4 grid)
        lines = formatted.strip().split("\n")
        assert len(lines) == 4

    def test_format_board_shows_revealed(self):
        """Test that revealed words are marked."""
        board_state = {
            "board": ["A", "B", "C", "D", "E", "F", "G", "H",
                     "I", "J", "K", "L", "M", "N", "O", "P"],
            "revealed": {letter: False for letter in "ABCDEFGHIJKLMNOP"},
        }
        board_state["revealed"]["A"] = True
        
        formatted = self.player._format_board_for_guesser(board_state)
        
        assert "[A]" in formatted  # Revealed words shown in brackets

