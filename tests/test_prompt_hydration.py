"""Tests for prompt template variable hydration."""

import random
import tempfile
from pathlib import Path

import pytest

from based.game import CodenamesGame
from based.player import HumanPlayer
from based.prompt_manager import PromptManager


class TestPromptHydration:
    """Test cases for prompt template variable replacement."""

    def setup_method(self):
        """Setup for each test."""
        # Set up a consistent game state for testing
        random.seed(42)
        
        # Create dummy players
        self.red_player = HumanPlayer()
        self.blue_player = HumanPlayer()
        
        # Initialize game
        self.game = CodenamesGame(
            words_file="inputs/names.yaml",
            red_player=self.red_player,
            blue_player=self.blue_player,
        )
        
        # Setup board with consistent seed
        self.game.setup_board()
        
        # Get board state
        self.board_state = self.game.get_board_state(reveal_all=True)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()

    def test_spymaster_prompt_hydration_red(self):
        """Test that red spymaster prompt variables are properly hydrated."""
        # Calculate context variables similar to CLI
        red_remaining = sum(
            1 for word, identity in self.board_state["identities"].items()
            if identity == "red_agent" and not self.board_state["revealed"].get(word, False)
        )
        blue_remaining = sum(
            1 for word, identity in self.board_state["identities"].items()
            if identity == "blue_agent" and not self.board_state["revealed"].get(word, False)
        )
        revealed_words = [word for word, revealed in self.board_state["revealed"].items() if revealed]
        
        # Categorize identities
        red_agents = [word for word, identity in self.board_state["identities"].items() 
                         if identity == "red_agent"]
        blue_agents = [word for word, identity in self.board_state["identities"].items() 
                          if identity == "blue_agent"]
        bystanders = [word for word, identity in self.board_state["identities"].items() 
                       if identity == "bystander"]
        assassin = [word for word, identity in self.board_state["identities"].items() 
                         if identity == "assassin"]
        
        # Load and format prompt
        context = {
            "board": self.board_state["board"],
            "revealed": self.board_state["revealed"],
            "team": "red",
            "red_remaining": red_remaining,
            "blue_remaining": blue_remaining,
            "revealed_words": ", ".join(revealed_words) if revealed_words else "None",
            "red_agents": ", ".join(red_agents),
            "blue_agents": ", ".join(blue_agents),
            "bystanders": ", ".join(bystanders),
            "assassin": ", ".join(assassin),
            "clue_history": "No previous clues yet",
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/red_spymaster.md", context)
        
        # Verify key variables are replaced (should not contain {{}} patterns)
        assert "{{RED_AGENTS}}" not in prompt, "RED_AGENTS variable not replaced"
        assert "{{BLUE_AGENTS}}" not in prompt, "BLUE_AGENTS variable not replaced"
        assert "{{BYSTANDERS}}" not in prompt, "BYSTANDERS variable not replaced"
        assert "{{ASSASSIN}}" not in prompt, "ASSASSIN variable not replaced"
        assert "{{CLUE_HISTORY}}" not in prompt, "CLUE_HISTORY variable not replaced"
        
        # Verify actual content is present
        if red_agents:
            assert red_agents[0] in prompt, f"Red agent {red_agents[0]} should be in prompt"
        if blue_agents:
            assert blue_agents[0] in prompt, f"Blue agent {blue_agents[0]} should be in prompt"
        if bystanders:
            assert bystanders[0] in prompt, f"Bystander {bystanders[0]} should be in prompt"
        if assassin:
            assert assassin[0] in prompt, f"Assassin {assassin[0]} should be in prompt"
        
        assert "No previous clues yet" in prompt, "Clue history should be present"

    def test_spymaster_prompt_hydration_blue(self):
        """Test that blue spymaster prompt variables are properly hydrated."""
        # Get blue agents
        blue_agents = [word for word, identity in self.board_state["identities"].items() 
                       if identity == "blue_agent"]
        red_agents = [word for word, identity in self.board_state["identities"].items() 
                      if identity == "red_agent"]
        
        context = {
            "team": "blue",
            "red_agents": ", ".join(red_agents),
            "blue_agents": ", ".join(blue_agents),
            "bystanders": "TEST_BYSTANDER",
            "assassin": "TEST_ASSASSIN",
            "clue_history": "Test history",
            "red_remaining": 8,
            "blue_remaining": 9,
            "revealed_words": "None",
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/blue_spymaster.md", context)
        
        # Verify no unreplaced variables
        assert "{{RED_AGENTS}}" not in prompt
        assert "{{BLUE_AGENTS}}" not in prompt
        assert "{{BYSTANDERS}}" not in prompt
        assert "{{ASSASSIN}}" not in prompt
        assert "{{CLUE_HISTORY}}" not in prompt
        
        # Verify blue-specific content
        if blue_agents:
            assert blue_agents[0] in prompt, "Blue team agents should be in blue spymaster prompt"

    def test_operative_prompt_hydration_red(self):
        """Test that red operative prompt variables are properly hydrated."""
        available_words = [
            word for word in self.board_state["board"] 
            if not self.board_state["revealed"].get(word, False)
        ]
        
        def _format_board_for_operative_cli(board_state):
            """Helper function to format board like CLI does."""
            board = board_state["board"]
            if len(board) != 25:
                return ", ".join(board)
            
            lines = []
            for row in range(5):
                row_items = board[row * 5 : (row + 1) * 5]
                lines.append(" | ".join(f"{item:>12}" for item in row_items))
            return "\n".join(lines)
        
        context = {
            "board": _format_board_for_operative_cli(self.board_state),
            "available_words": ", ".join(available_words),
            "clue_history": "None (game just started)",
            "clue": "ANIMALS",
            "number": 2,
            "team": "red",
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/red_operative.md", context)
        
        # Verify key variables are replaced
        assert "{{BOARD}}" not in prompt, "BOARD variable not replaced"
        assert "{{AVAILABLE_WORDS}}" not in prompt, "AVAILABLE_WORDS variable not replaced"
        assert "{{CLUE_HISTORY}}" not in prompt, "CLUE_HISTORY variable not replaced"
        assert "{{CLUE}}" not in prompt, "CLUE variable not replaced"
        assert "{{NUMBER}}" not in prompt, "NUMBER variable not replaced"
        
        # Verify actual content is present
        assert "ANIMALS" in prompt, "Clue should be in prompt"
        assert "2" in prompt, "Number should be in prompt"
        assert "None (game just started)" in prompt, "Clue history should be in prompt"
        if available_words:
            assert available_words[0] in prompt, f"Available word {available_words[0]} should be in prompt"

    def test_operative_prompt_hydration_blue(self):
        """Test that blue operative prompt variables are properly hydrated."""
        context = {
            "board": "TEST_BOARD",
            "available_words": "WORD1, WORD2, WORD3",
            "clue_history": "Previous clue: TOOLS (2)",
            "clue": "WEAPONS",
            "number": 3,
            "team": "blue",
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/blue_operative.md", context)
        
        # Verify no unreplaced variables
        assert "{{BOARD}}" not in prompt
        assert "{{AVAILABLE_WORDS}}" not in prompt
        assert "{{CLUE_HISTORY}}" not in prompt
        assert "{{CLUE}}" not in prompt
        assert "{{NUMBER}}" not in prompt
        
        # Verify content
        assert "WEAPONS" in prompt, "Clue should be in prompt"
        assert "3" in prompt, "Number should be in prompt"
        assert "Previous clue: TOOLS (2)" in prompt, "Clue history should be in prompt"
        assert "WORD1, WORD2, WORD3" in prompt, "Available words should be in prompt"

    def test_operative_prompt_special_numbers(self):
        """Test that operative prompts handle special number values (0, unlimited)."""
        # Test with 0
        context = {
            "board": "TEST_BOARD",
            "available_words": "WORD1, WORD2",
            "clue_history": "No history",
            "clue": "ZERO_TEST",
            "number": 0,
            "team": "red",
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/red_operative.md", context)
        assert "0" in prompt, "Zero should be displayed in prompt"
        assert "ZERO_TEST" in prompt, "Clue should be in prompt"
        
        # Test with unlimited
        context["number"] = "unlimited"
        context["clue"] = "UNLIMITED_TEST"
        
        prompt = self.prompt_manager.load_prompt("prompts/red_operative.md", context)
        assert "unlimited" in prompt, "Unlimited should be displayed in prompt"
        assert "UNLIMITED_TEST" in prompt, "Clue should be in prompt"

    def test_referee_prompt_hydration(self):
        """Test that referee prompt variables are properly hydrated."""
        # Get team agents
        team_agents = [
            word for word, identity in self.board_state["identities"].items()
            if identity == "red_agent"
        ]
        
        context = {
            "clue": "EXAMPLE",
            "number": 2,
            "team": "red",
            "board": ", ".join(self.board_state["board"]),
            "team_agents": ", ".join(team_agents),
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/referee.md", context)
        
        # Verify key variables are replaced
        assert "{{CLUE}}" not in prompt, "CLUE variable not replaced"
        assert "{{NUMBER}}" not in prompt, "NUMBER variable not replaced"
        assert "{{TEAM}}" not in prompt, "TEAM variable not replaced"
        assert "{{BOARD}}" not in prompt, "BOARD variable not replaced"
        assert "{{TEAM_AGENTS}}" not in prompt, "TEAM_AGENTS variable not replaced"
        
        # Verify actual content is present
        assert "EXAMPLE" in prompt, "Clue should be in prompt"
        assert "2" in prompt, "Number should be in prompt"
        assert "red" in prompt, "Team should be in prompt"
        
        # Verify board words are present
        if self.board_state["board"]:
            assert self.board_state["board"][0] in prompt, "Board words should be in prompt"
        
        # Verify team agents are present
        if team_agents:
            assert team_agents[0] in prompt, "Team agents should be in prompt"

    def test_referee_prompt_special_cases(self):
        """Test referee prompt with special number values and edge cases."""
        context = {
            "clue": "SPECIAL_CASE",
            "number": "unlimited",
            "team": "blue",
            "board": "WORD1, WORD2, WORD3",
            "team_agents": "AGENT1, AGENT2",
        }
        
        prompt = self.prompt_manager.load_prompt("prompts/referee.md", context)
        
        # Verify special values are handled
        assert "unlimited" in prompt, "Unlimited should be in prompt"
        assert "SPECIAL_CASE" in prompt, "Special case clue should be in prompt"
        assert "blue" in prompt, "Blue team should be in prompt"

    def test_prompt_includes_work(self):
        """Test that {{include}} directives are processed correctly."""
        # All our prompts include shared/game_rules.md
        context = {"test": "value"}
        
        spymaster_prompt = self.prompt_manager.load_prompt("prompts/red_spymaster.md", context)
        operative_prompt = self.prompt_manager.load_prompt("prompts/red_operative.md", context)
        referee_prompt = self.prompt_manager.load_prompt("prompts/referee.md", context)
        
        # Check that game rules content appears in all prompts
        game_rules_content = "Codenames"
        
        assert game_rules_content in spymaster_prompt, "Game rules should be included in spymaster prompt"
        assert game_rules_content in operative_prompt, "Game rules should be included in operative prompt"
        assert game_rules_content in referee_prompt, "Game rules should be included in referee prompt"
        
        # Verify {{include}} directive is removed
        assert "{{include:" not in spymaster_prompt, "Include directive should be processed"
        assert "{{include:" not in operative_prompt, "Include directive should be processed"
        assert "{{include:" not in referee_prompt, "Include directive should be processed"

    def test_missing_variables_handled_gracefully(self):
        """Test that missing context variables don't break prompt generation."""
        # Minimal context (missing some expected variables)
        context = {
            "team": "red",
            "clue": "TEST",
        }
        
        # Should not raise exceptions even with missing variables
        spymaster_prompt = self.prompt_manager.load_prompt("prompts/red_spymaster.md", context)
        operative_prompt = self.prompt_manager.load_prompt("prompts/red_operative.md", context)
        referee_prompt = self.prompt_manager.load_prompt("prompts/referee.md", context)
        
        # Should still contain basic content
        assert "Red Team Spymaster" in spymaster_prompt
        assert "Red Team Operatives" in operative_prompt
        assert "Referee" in referee_prompt
        
        # Missing variables should remain as placeholders
        assert "{{RED_AGENTS}}" in spymaster_prompt  # Missing variable
        assert "{{BOARD}}" in operative_prompt  # Missing variable
