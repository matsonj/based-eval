#!/usr/bin/env python3
"""Demo of human vs human Codenames gameplay."""

import random

from based.game import CodenamesGame
from based.player import HumanPlayer


# Create a simple demo that doesn't require API keys
def demo_human_vs_human():
    """Run a demonstration of human vs human gameplay."""
    print("🎯 BASED Eval - Codenames Demo")
    print("=" * 50)

    # Set seed for reproducible demo
    random.seed(42)

    # Create human players
    red_player = HumanPlayer()
    blue_player = HumanPlayer()

    # Create game
    game = CodenamesGame(
        words_file="inputs/names.yaml",
        red_player=red_player,
        blue_player=blue_player,
    )

    print("Setting up board...")
    game.setup_board()

    print("\nBoard created with hidden identities!")
    print("In a real game, only the Spymaster would see all identities.")
    print("\nHere's the board with all identities revealed (for demo purposes):")

    game.display_board(reveal_all=True)

    print("\nIdentity Summary:")
    red_agents = [
        word
        for word, identity in game.identities.items()
        if identity == "red_agent"
    ]
    blue_agents = [
        word
        for word, identity in game.identities.items()
        if identity == "blue_agent"
    ]
    bystanders = [
        word for word, identity in game.identities.items() if identity == "bystander"
    ]
    assassin = [word for word, identity in game.identities.items() if identity == "assassin"][0]

    print(f"🔴 Red Agents ({len(red_agents)}): {', '.join(red_agents)}")
    print(f"🔵 Blue Agents ({len(blue_agents)}): {', '.join(blue_agents)}")
    print(f"😐 Bystanders ({len(bystanders)}): {', '.join(bystanders)}")
    print(f"💀 Assassin: {assassin}")

    print("\n" + "=" * 50)
    print("In a real game, the Operatives would only see unrevealed words")
    print("and receive clues from their Spymasters.")
    print("Demo complete!")


if __name__ == "__main__":
    demo_human_vs_human()
