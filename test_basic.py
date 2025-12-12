#!/usr/bin/env python3
"""Basic test of Codenames game setup without AI calls."""

import random

from codenames.game import CodenamesGame
from codenames.player import HumanPlayer


# Mock human player for testing
class MockHumanPlayer(HumanPlayer):
    def __init__(self, moves=None):
        self.moves = moves or []
        self.move_index = 0

    def get_next_move(self):
        if self.move_index < len(self.moves):
            move = self.moves[self.move_index]
            self.move_index += 1
            return move
        return "ALPHA"  # Default fallback


def test_game_setup():
    """Test basic game setup and board generation."""
    print("Testing Codenames game setup...")

    # Set seed for reproducible test
    random.seed(42)

    # Create mock players
    red_player = MockHumanPlayer()
    blue_player = MockHumanPlayer()

    # Create game
    game = CodenamesGame(
        words_file="inputs/names.yaml",
        red_player=red_player,
        blue_player=blue_player,
    )

    # Test board setup
    game.setup_board()

    print(f"Board size: {len(game.board)}")
    print(f"Board: {game.board[:10]}...")  # Show first 10 words

    # Count identities
    red_count = sum(
        1 for identity in game.identities.values() if identity == "red_agent"
    )
    blue_count = sum(
        1 for identity in game.identities.values() if identity == "blue_agent"
    )
    bystander_count = sum(
        1 for identity in game.identities.values() if identity == "bystander"
    )
    assassin_count = sum(1 for identity in game.identities.values() if identity == "assassin")

    print(f"Red agents: {red_count}")
    print(f"Blue agents: {blue_count}")
    print(f"Bystanders: {bystander_count}")
    print(f"Assassin: {assassin_count}")

    # Verify counts
    assert len(game.board) == 25, f"Expected 25 words, got {len(game.board)}"
    assert red_count == 9, f"Expected 9 red agents, got {red_count}"
    assert blue_count == 8, f"Expected 8 blue agents, got {blue_count}"
    assert bystander_count == 7, f"Expected 7 bystanders, got {bystander_count}"
    assert assassin_count == 1, f"Expected 1 assassin, got {assassin_count}"

    print("✓ Game setup test passed!")

    # Test board display
    print("\nTesting board display...")
    game.display_board(reveal_all=True)

    print("✓ Board display test passed!")


if __name__ == "__main__":
    test_game_setup()
