"""Tests for the core Codenames game logic."""

import random

import pytest

from based.game import CodenamesGame
from based.player import HumanPlayer


class MockHumanPlayer(HumanPlayer):
    """Mock human player for testing."""

    def __init__(self, moves=None):
        self.moves = moves or []
        self.move_index = 0

    def get_next_move(self):
        if self.move_index < len(self.moves):
            move = self.moves[self.move_index]
            self.move_index += 1
            return move
        return "ALPHA"  # Default fallback


class TestCodenamesGame:
    """Test cases for CodenamesGame."""

    def setup_method(self):
        """Setup for each test."""
        random.seed(42)  # Reproducible tests
        self.red_player = MockHumanPlayer()
        self.blue_player = MockHumanPlayer()
        self.game = CodenamesGame(
            words_file="inputs/names.yaml",
            red_player=self.red_player,
            blue_player=self.blue_player,
        )

    def test_board_setup(self):
        """Test board initialization."""
        self.game.setup_board()

        # Check board size
        assert len(self.game.board) == 25

        # Check identity counts
        red_count = sum(
            1
            for identity in self.game.identities.values()
            if identity == "red_agent"
        )
        blue_count = sum(
            1
            for identity in self.game.identities.values()
            if identity == "blue_agent"
        )
        bystander_count = sum(
            1 for identity in self.game.identities.values() if identity == "bystander"
        )
        assassin_count = sum(
            1 for identity in self.game.identities.values() if identity == "assassin"
        )

        assert red_count == 9
        assert blue_count == 8
        assert bystander_count == 7
        assert assassin_count == 1

        # Check all words are initially unrevealed
        assert all(not revealed for revealed in self.game.revealed.values())

    def test_board_state(self):
        """Test board state retrieval."""
        self.game.setup_board()

        # Test public board state
        public_state = self.game.get_board_state(reveal_all=False)
        assert len(public_state["board"]) == 25
        assert public_state["current_team"] == "red"
        assert public_state["turn_count"] == 0
        assert len(public_state["identities"]) == 0  # Nothing revealed yet

        # Test revealed board state
        full_state = self.game.get_board_state(reveal_all=True)
        assert len(full_state["identities"]) == 25  # All identities shown

    def test_process_guess_correct(self):
        """Test processing a correct guess."""
        self.game.setup_board()

        # Find an agent of the current team to guess
        current_team_agent = None
        for word, identity in self.game.identities.items():
            if identity == f"{self.game.current_team}_agent":
                current_team_agent = word
                break

        assert current_team_agent is not None

        # Process the guess
        result = self.game.process_guess(current_team_agent)

        assert result is True
        assert self.game.revealed[current_team_agent] is True
        assert len(self.game.moves_log) == 1
        assert self.game.moves_log[0]["correct"] is True

    def test_process_guess_bystander(self):
        """Test processing a bystander guess."""
        self.game.setup_board()

        # Find a bystander to guess
        bystander = None
        for word, identity in self.game.identities.items():
            if identity == "bystander":
                bystander = word
                break

        assert bystander is not None

        # Process the guess
        result = self.game.process_guess(bystander)

        assert result is False
        assert self.game.revealed[bystander] is True
        assert len(self.game.moves_log) == 1
        assert self.game.moves_log[0]["correct"] is False

    def test_process_guess_assassin(self):
        """Test processing an assassin guess (instant loss)."""
        self.game.setup_board()

        # Find the assassin
        assassin = None
        for word, identity in self.game.identities.items():
            if identity == "assassin":
                assassin = word
                break

        assert assassin is not None

        # Process the guess
        result = self.game.process_guess(assassin)

        assert result is False
        assert self.game.game_over is True
        assert self.game.winner == "blue"  # Red team loses
        assert self.game.revealed[assassin] is True

    def test_switch_teams(self):
        """Test team switching."""
        assert self.game.current_team == "red"
        assert self.game.turn_count == 0

        self.game.switch_teams()

        assert self.game.current_team == "blue"
        assert self.game.turn_count == 1

        self.game.switch_teams()

        assert self.game.current_team == "red"
        assert self.game.turn_count == 2

    def test_win_condition(self):
        """Test win condition detection."""
        self.game.setup_board()

        # Reveal all red agents except one
        red_agents = [
            word
            for word, identity in self.game.identities.items()
            if identity == "red_agent"
        ]

        # Reveal all but the last one
        for word in red_agents[:-1]:
            self.game.revealed[word] = True

        # Process the last red agent
        result = self.game.process_guess(red_agents[-1])

        assert result is True
        assert self.game.game_over is True
        assert self.game.winner == "red"
