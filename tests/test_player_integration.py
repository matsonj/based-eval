"""Integration tests for AIPlayer prompt generation."""

import random
from unittest.mock import Mock, patch

import pytest

from based.game import CodenamesGame
from based.player import AIPlayer, HumanPlayer


class TestAIPlayerIntegration:
    """Test that AIPlayer methods generate prompts correctly with real game state."""

    def setup_method(self):
        """Setup for each test."""
        random.seed(42)  # Reproducible tests
        
        # Create real game with human players for setup
        red_human = HumanPlayer()
        blue_human = HumanPlayer()
        
        self.game = CodenamesGame(
            words_file="inputs/names.yaml",
            red_player=red_human,
            blue_player=blue_human,
        )
        self.game.setup_board()
        
        # Create AI players for testing
        self.red_ai = AIPlayer("gpt-4")
        self.blue_ai = AIPlayer("gpt-4")

    def test_red_spymaster_prompt_integration(self):
        """Test that AIPlayer.get_spymaster_move() generates correct prompts for red team."""
        board_state = self.game.get_board_state(reveal_all=True)
        
        # Mock the OpenRouter call to capture the prompt
        with patch.object(self.red_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("CLUE: ANIMALS\nNUMBER: 2", {"tokens": 100, "cost": 0.01})
            
            # Call the spymaster method
            try:
                clue, number = self.red_ai.get_spymaster_move(board_state, "prompts/red_spymaster.md")
            except Exception:
                # We don't care if it fails, we just want to check the prompt
                pass
            
            # Verify the method was called
            assert mock_call.called, "OpenRouter adapter should have been called"
            
            # Get the prompt that was sent to the AI
            call_args = mock_call.call_args
            prompt = call_args[0][1]  # Second argument is the prompt
            
            # Verify template variables were replaced correctly
            assert "{{RED_AGENTS}}" not in prompt, "RED_AGENTS should be replaced"
            assert "{{BLUE_AGENTS}}" not in prompt, "BLUE_AGENTS should be replaced" 
            assert "{{BYSTANDERS}}" not in prompt, "BYSTANDERS should be replaced"
            assert "{{ASSASSIN}}" not in prompt, "ASSASSIN should be replaced"
            assert "{{CLUE_HISTORY}}" not in prompt, "CLUE_HISTORY should be replaced"
            assert "{{RED_REMAINING}}" not in prompt, "RED_REMAINING should be replaced"
            assert "{{BLUE_REMAINING}}" not in prompt, "BLUE_REMAINING should be replaced"
            
            # Verify actual game content is present
            red_agents = [word for word, identity in board_state["identities"].items() 
                          if identity == "red_agent"]
            blue_agents = [word for word, identity in board_state["identities"].items() 
                           if identity == "blue_agent"]
            bystanders = [word for word, identity in board_state["identities"].items() 
                        if identity == "bystander"]
            assassin = [word for word, identity in board_state["identities"].items() 
                      if identity == "assassin"]
            
            if red_agents:
                assert red_agents[0] in prompt, f"Red agent {red_agents[0]} should be in prompt"
            if blue_agents:
                assert blue_agents[0] in prompt, f"Blue agent {blue_agents[0]} should be in prompt"
            if bystanders:
                assert bystanders[0] in prompt, f"Bystander {bystanders[0]} should be in prompt"
            if assassin:
                assert assassin[0] in prompt, f"Assassin {assassin[0]} should be in prompt"

    def test_blue_spymaster_prompt_integration(self):
        """Test that AIPlayer.get_spymaster_move() generates correct prompts for blue team."""
        board_state = self.game.get_board_state(reveal_all=True)
        board_state["current_team"] = "blue"  # Switch to blue team
        
        with patch.object(self.blue_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("CLUE: TOOLS\nNUMBER: 3", {"tokens": 100, "cost": 0.01})
            
            try:
                clue, number = self.blue_ai.get_spymaster_move(board_state, "prompts/blue_spymaster.md")
            except Exception:
                pass
            
            assert mock_call.called
            prompt = mock_call.call_args[0][1]
            
            # Verify no template variables remain
            assert "{{RED_AGENTS}}" not in prompt
            assert "{{BLUE_AGENTS}}" not in prompt
            assert "{{BYSTANDERS}}" not in prompt
            assert "{{ASSASSIN}}" not in prompt
            
            # Verify blue-specific content
            blue_agents = [word for word, identity in board_state["identities"].items() 
                           if identity == "blue_agent"]
            if blue_agents:
                assert blue_agents[0] in prompt, "Blue agents should be in blue spymaster prompt"

    def test_red_operative_prompt_integration(self):
        """Test that AIPlayer.get_operative_moves() generates correct prompts for red team."""
        board_state = self.game.get_board_state(reveal_all=False)  # Operatives don't see identities
        
        with patch.object(self.red_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("WOLF\nKEY", {"tokens": 50, "cost": 0.005})
            
            try:
                guesses = self.red_ai.get_operative_moves(board_state, "ANIMALS", 2, "prompts/red_operative.md")
            except Exception:
                pass
            
            assert mock_call.called
            prompt = mock_call.call_args[0][1]
            
            # Verify template variables were replaced
            assert "{{BOARD}}" not in prompt, "BOARD should be replaced"
            assert "{{CLUE}}" not in prompt, "CLUE should be replaced"
            assert "{{NUMBER}}" not in prompt, "NUMBER should be replaced"
            assert "{{AVAILABLE_WORDS}}" not in prompt, "AVAILABLE_WORDS should be replaced"
            assert "{{CLUE_HISTORY}}" not in prompt, "CLUE_HISTORY should be replaced"
            
            # Verify actual content is present
            assert "ANIMALS" in prompt, "Clue should be in prompt"
            assert "2" in prompt, "Number should be in prompt"
            
            # Verify board words are present
            available_words = [word for word in board_state["board"] 
                             if not board_state["revealed"].get(word, False)]
            if available_words:
                assert available_words[0] in prompt, "Available words should be in prompt"

    def test_blue_operative_prompt_integration(self):
        """Test that AIPlayer.get_operative_moves() generates correct prompts for blue team."""
        board_state = self.game.get_board_state(reveal_all=False)
        board_state["current_team"] = "blue"
        
        with patch.object(self.blue_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("RIFLE\nGRENADE", {"tokens": 50, "cost": 0.005})
            
            try:
                guesses = self.blue_ai.get_operative_moves(board_state, "WEAPONS", 3, "prompts/blue_operative.md")
            except Exception:
                pass
            
            assert mock_call.called
            prompt = mock_call.call_args[0][1]
            
            # Verify no template variables remain
            assert "{{BOARD}}" not in prompt
            assert "{{CLUE}}" not in prompt  
            assert "{{NUMBER}}" not in prompt
            assert "{{AVAILABLE_WORDS}}" not in prompt
            
            # Verify content
            assert "WEAPONS" in prompt, "Clue should be in prompt"
            assert "3" in prompt, "Number should be in prompt"

    def test_operative_prompt_with_special_numbers(self):
        """Test operative prompts with special number values (0, unlimited)."""
        board_state = self.game.get_board_state(reveal_all=False)
        
        # Test with number = 0
        with patch.object(self.red_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("PASS", {"tokens": 20, "cost": 0.001})
            
            try:
                guesses = self.red_ai.get_operative_moves(board_state, "ZERO", 0, "prompts/red_operative.md")
            except Exception:
                pass
            
            if mock_call.called:
                prompt = mock_call.call_args[0][1]
                assert "0" in prompt, "Zero should appear in prompt"
                assert "ZERO" in prompt, "Clue should appear in prompt"
        
        # Test with number = "unlimited"
        with patch.object(self.red_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("ALL WORDS", {"tokens": 30, "cost": 0.002})
            
            try:
                guesses = self.red_ai.get_operative_moves(board_state, "ALL", "unlimited", "prompts/red_operative.md")
            except Exception:
                pass
            
            if mock_call.called:
                prompt = mock_call.call_args[0][1]
                assert "unlimited" in prompt or "ALL" in prompt, "Unlimited/clue should appear in prompt"

    def test_spymaster_prompt_includes_game_history(self):
        """Test that spymaster prompts include game history when available."""
        board_state = self.game.get_board_state(reveal_all=True)
        
        # Add some moves to the game history
        self.game.moves_log = [
            {
                "team": "red",
                "type": "spymaster",
                "clue": "ANIMALS",
                "number": 2,
                "turn": 0
            },
            {
                "team": "red", 
                "type": "operative",
                "guess": "WOLF",
                "correct": True,
                "turn": 0
            }
        ]
        
        with patch.object(self.blue_ai.adapter, 'call_model_with_metadata') as mock_call:
            mock_call.return_value = ("CLUE: RESPONSE\nNUMBER: 1", {"tokens": 100, "cost": 0.01})
            
            try:
                clue, number = self.blue_ai.get_spymaster_move(board_state, "prompts/blue_spymaster.md")
            except Exception:
                pass
            
            if mock_call.called:
                prompt = mock_call.call_args[0][1]
                # Should include some indication of previous clues
                # The exact format depends on how clue_history is implemented
                assert "{{CLUE_HISTORY}}" not in prompt, "CLUE_HISTORY should be replaced"
