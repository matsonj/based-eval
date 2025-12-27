"""Tests for the ChainLex-1 game logic."""

import random
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest

from chainlex.game import ChainLexGame
from chainlex.player import AIPlayer


class MockAIPlayer:
    """Mock AI player for testing."""

    def __init__(self, model_name: str = "test-model", clue: str = "TEST", number: int = 3, guesses: list = None):
        self.model_name = model_name
        self._clue = clue
        self._number = number
        self._guesses = guesses or ["WORD1", "WORD2"]
        self._adapter = Mock()
        self._adapter.resolve_model.return_value = model_name
        self._last_call_metadata = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "latency_ms": 500,
            "openrouter_cost": 0.005,
            "upstream_cost": 0.003,
            "request_text": "Test prompt",
            "response_text": "Test response",
        }

    @property
    def adapter(self):
        return self._adapter

    def get_clue_giver_move(self, board_state, prompt_file, head_to_head_context=""):
        return self._clue, self._number

    def get_guesser_moves(self, board_state, clue, number, prompt_file, head_to_head_context=""):
        # Return only words that are on the board
        board_words = set(board_state["board"])
        return [g for g in self._guesses if g in board_words][:number + 1]

    def get_last_call_metadata(self):
        return self._last_call_metadata


class TestChainLexGameSetup:
    """Test cases for ChainLexGame board setup."""

    def setup_method(self):
        """Setup for each test."""
        random.seed(42)
        self.player_away = MockAIPlayer("model-away")
        self.player_home = MockAIPlayer("model-home")
        self.game = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=self.player_away,
            player_home=self.player_home,
            quiet=True,
            seed=42,
        )

    def test_board_setup_size(self):
        """Test that board has correct size."""
        self.game.setup_board()
        assert len(self.game.board) == 16

    def test_board_setup_identities(self):
        """Test that identities are correctly distributed."""
        self.game.setup_board()

        friendly_count = sum(
            1 for identity in self.game.identities.values()
            if identity == "friendly"
        )
        bystander_count = sum(
            1 for identity in self.game.identities.values()
            if identity == "bystander"
        )
        assassin_count = sum(
            1 for identity in self.game.identities.values()
            if identity == "assassin"
        )

        assert friendly_count == 8, "Should have 8 friendly words"
        assert bystander_count == 7, "Should have 7 bystanders"
        assert assassin_count == 1, "Should have 1 assassin"

    def test_board_setup_all_unrevealed(self):
        """Test that all words start unrevealed."""
        self.game.setup_board()
        assert all(not revealed for revealed in self.game.revealed.values())

    def test_board_setup_reproducibility(self):
        """Test that same seed produces same board."""
        self.game.setup_board()
        board1 = self.game.board.copy()
        identities1 = self.game.identities.copy()

        # Create new game with same seed
        game2 = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=self.player_away,
            player_home=self.player_home,
            quiet=True,
            seed=42,
        )
        game2.setup_board()

        assert board1 == game2.board
        assert identities1 == game2.identities


class TestChainLexScoring:
    """Test cases for ChainLex-1 scoring logic."""

    def test_max_possible_score(self):
        """Test maximum possible score calculation (triangular: 1+2+3+...+8 = 36)."""
        game = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=MockAIPlayer(),
            player_home=MockAIPlayer(),
            quiet=True,
        )
        assert game.calculate_max_possible_score() == 36

    def test_triangular_scoring_formula(self):
        """Test that scoring follows triangular number formula."""
        # n(n+1)/2 for 8 friendly words = 8*9/2 = 36
        n = 8
        expected = n * (n + 1) // 2
        assert expected == 36


class TestChainLexClueValidation:
    """Test cases for clue validation."""

    def setup_method(self):
        """Setup for each test."""
        self.game = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=MockAIPlayer(),
            player_home=MockAIPlayer(),
            quiet=True,
        )
        self.game.setup_board()
        self.board_state = self.game.get_board_state(reveal_all=True)

    def test_valid_clue(self):
        """Test that valid clues pass validation."""
        clue, number, is_valid, reasoning = self.game._validate_clue(
            "WATER", 3, self.board_state
        )
        assert is_valid is True

    def test_invalid_clue_multiple_words(self):
        """Test that multi-word clues are rejected."""
        clue, number, is_valid, reasoning = self.game._validate_clue(
            "TWO WORDS", 2, self.board_state
        )
        assert is_valid is False
        assert "Multiple words" in reasoning

    def test_invalid_clue_matches_board_word(self):
        """Test that clue matching a board word is rejected."""
        board_word = self.board_state["board"][0]
        clue, number, is_valid, reasoning = self.game._validate_clue(
            board_word, 2, self.board_state
        )
        assert is_valid is False
        assert "matches word on board" in reasoning


class TestChainLexBoardState:
    """Test cases for board state management."""

    def setup_method(self):
        """Setup for each test."""
        random.seed(42)
        self.game = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=MockAIPlayer(),
            player_home=MockAIPlayer(),
            quiet=True,
            seed=42,
        )
        self.game.setup_board()

    def test_board_state_public_view(self):
        """Test public board state doesn't reveal identities."""
        state = self.game.get_board_state(reveal_all=False)
        assert len(state["board"]) == 16
        assert len(state["identities"]) == 0  # No identities revealed

    def test_board_state_full_view(self):
        """Test full board state reveals all identities."""
        state = self.game.get_board_state(reveal_all=True)
        assert len(state["board"]) == 16
        assert len(state["identities"]) == 16  # All identities shown

    def test_board_state_partial_reveal(self):
        """Test board state with some words revealed."""
        revealed = {word: False for word in self.game.board}
        revealed[self.game.board[0]] = True
        
        state = self.game.get_board_state(reveal_all=False, revealed=revealed)
        assert len(state["identities"]) == 1  # Only one revealed


class TestChainLexControllog:
    """Test cases for controllog integration."""

    def setup_method(self):
        """Setup for each test."""
        random.seed(42)
        self.player_away = MockAIPlayer("model-away")
        self.player_home = MockAIPlayer("model-home")
        self.game = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=self.player_away,
            player_home=self.player_home,
            quiet=True,
            seed=42,
        )

    def test_controllog_initialization(self):
        """Test controllog initializes correctly."""
        with patch('chainlex.game.cl') as mock_cl:
            self.game.init_controllog(Path("/tmp/test"), "test_run_123")
            
            mock_cl.init.assert_called_once()
            assert self.game._controllog_initialized is True
            assert self.game._run_id == "test_run_123"

    def test_emit_state_move(self):
        """Test state transition events are emitted."""
        with patch('chainlex.game.cl') as mock_cl:
            self.game.init_controllog(Path("/tmp/test"), "test_run")
            self.game._emit_state_move("NEW", "WIP", {"test": "payload"})
            
            mock_cl.state_move.assert_called_once()
            call_kwargs = mock_cl.state_move.call_args[1]
            assert call_kwargs["from_"] == "NEW"
            assert call_kwargs["to"] == "WIP"

    def test_emit_model_events_with_text(self):
        """Test model events include request/response text."""
        with patch('chainlex.game.cl') as mock_cl:
            self.game.init_controllog(Path("/tmp/test"), "test_run")
            
            self.game._emit_model_events(
                player=self.player_away,
                call_type="clue_giver",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=1000,
                cost=0.005,
                upstream_cost=0.003,
                request_text="Test prompt content",
                response_text="Test response content",
                payload={"clue": "TEST"},
            )
            
            # Verify model_prompt was called with request_text
            mock_cl.model_prompt.assert_called_once()
            prompt_kwargs = mock_cl.model_prompt.call_args[1]
            assert prompt_kwargs["request_text"] == "Test prompt content"
            
            # Verify model_completion was called with response_text
            mock_cl.model_completion.assert_called_once()
            completion_kwargs = mock_cl.model_completion.call_args[1]
            assert completion_kwargs["response_text"] == "Test response content"

    def test_emit_game_complete(self):
        """Test game_complete event is emitted with correct payload."""
        with patch('chainlex.game.cl') as mock_cl:
            self.game.init_controllog(Path("/tmp/test"), "test_run")
            
            self.game._emit_game_complete(
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
                cost_money=0.01,
                upstream_cost_money=0.007,
            )
            
            mock_cl.game_complete.assert_called_once()
            call_kwargs = mock_cl.game_complete.call_args[1]
            
            assert call_kwargs["model_away"] == "claude-3"
            assert call_kwargs["model_home"] == "gpt-4"
            assert call_kwargs["outcome"] == "model_away"
            assert call_kwargs["winner_model"] == "claude-3"
            assert call_kwargs["score_away"] == 15
            assert call_kwargs["score_home"] == 10
            assert call_kwargs["margin"] == 5
            assert call_kwargs["total_guesses"] == 8
            assert call_kwargs["wall_ms"] == 45000


class TestChainLexGameId:
    """Test cases for game ID generation."""

    def test_game_id_generated(self):
        """Test that game ID is generated on creation."""
        game = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=MockAIPlayer(),
            player_home=MockAIPlayer(),
            quiet=True,
        )
        assert game.game_id is not None
        assert len(game.game_id) == 8  # UUID prefix

    def test_game_id_unique(self):
        """Test that each game gets a unique ID."""
        game1 = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=MockAIPlayer(),
            player_home=MockAIPlayer(),
            quiet=True,
        )
        game2 = ChainLexGame(
            words_file="inputs/names.yaml",
            player_away=MockAIPlayer(),
            player_home=MockAIPlayer(),
            quiet=True,
        )
        assert game1.game_id != game2.game_id


class TestChainLexVersion:
    """Test cases for game versioning."""

    def test_version_constant(self):
        """Test that VERSION constant is defined."""
        assert hasattr(ChainLexGame, 'VERSION')
        assert ChainLexGame.VERSION == "1.0.0"


class TestChainLexConstants:
    """Test cases for game constants."""

    def test_board_size(self):
        """Test BOARD_SIZE constant."""
        assert ChainLexGame.BOARD_SIZE == 16

    def test_friendly_words_count(self):
        """Test FRIENDLY_WORDS constant."""
        assert ChainLexGame.FRIENDLY_WORDS == 8

    def test_bystanders_count(self):
        """Test BYSTANDERS constant."""
        assert ChainLexGame.BYSTANDERS == 7

    def test_assassins_count(self):
        """Test ASSASSINS constant."""
        assert ChainLexGame.ASSASSINS == 1

    def test_penalty_values(self):
        """Test penalty constants."""
        assert ChainLexGame.BYSTANDER_PENALTY == -5
        assert ChainLexGame.ASSASSIN_PENALTY == -28

