"""Player classes for BASED eval - Codenames game."""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple

from codenames.adapters.openrouter_adapter import OpenRouterAdapter
from codenames.prompt_manager import PromptManager
from codenames.utils.logging import log_ai_call_metadata, format_turn_label

logger = logging.getLogger(__name__)


class Player(ABC):
    """Abstract base class for all players."""

    @abstractmethod
    def get_spymaster_move(self, board_state: Dict, prompt_file: str) -> Tuple[str, int|str]:
        """Get clue and number from spymaster."""
        pass

    @abstractmethod
    def get_operative_moves(
        self, board_state: Dict, clue: str, number: int|str, prompt_file: str
    ) -> List[str]:
        """Get guesses from operative."""
        pass


class HumanPlayer(Player):
    """Human player implementation."""

    def get_spymaster_move(self, board_state: Dict, prompt_file: str) -> Tuple[str, int|str]:
        """Human spymaster input is handled in the game loop."""
        raise NotImplementedError("Human spymaster input handled in game loop")

    def get_operative_moves(
        self, board_state: Dict, clue: str, number: int|str, prompt_file: str
    ) -> List[str]:
        """Human operative input is handled in the game loop."""
        raise NotImplementedError("Human operative input handled in game loop")


class AIPlayer(Player):
    """AI player using OpenRouter models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._adapter = None
        self.prompt_manager = PromptManager()
        self._last_call_metadata = None

        logger.info(f"Created AI player with model: {model_name}")

    @property
    def adapter(self):
        """Lazy initialization of OpenRouter adapter."""
        if self._adapter is None:
            self._adapter = OpenRouterAdapter()
        return self._adapter

    def get_last_call_metadata(self):
        """Get metadata from the last AI call."""
        return self._last_call_metadata

    def get_spymaster_move(self, board_state: Dict, prompt_file: str) -> Tuple[str, int|str]:
        """Get clue and number from AI spymaster."""
        return self._get_spymaster_move_with_retry(board_state, prompt_file, is_retry=False)

    def _get_spymaster_move_with_retry(self, board_state: Dict, prompt_file: str, is_retry: bool) -> Tuple[str, int|str]:
        """Internal method to get spymaster move with retry tracking."""
        try:
            # Calculate remaining agents
            red_remaining = sum(
                1 for word, identity in board_state["identities"].items()
                if identity == "red_agent" and not board_state["revealed"].get(word, False)
            )
            blue_remaining = sum(
                1 for word, identity in board_state["identities"].items()
                if identity == "blue_agent" and not board_state["revealed"].get(word, False)
            )
            revealed_words = [word for word, revealed in board_state["revealed"].items() if revealed]
            
            # Categorize identities for cleaner prompt formatting
            red_agents = [word for word, identity in board_state["identities"].items() 
                             if identity == "red_agent"]
            blue_agents = [word for word, identity in board_state["identities"].items() 
                              if identity == "blue_agent"]
            bystanders = [word for word, identity in board_state["identities"].items() 
                        if identity == "bystander"]
            assassin = [word for word, identity in board_state["identities"].items() 
                   if identity == "assassin"]
            
            # Load and format prompt
            prompt = self.prompt_manager.load_prompt(
                prompt_file,
                {
                    "board": board_state["board"],
                    "revealed": ", ".join(revealed_words) if revealed_words else "None",
                    "team": board_state["current_team"],
                    "red_remaining": red_remaining,
                    "blue_remaining": blue_remaining,
                    "red_agents": ", ".join(red_agents),
                    "blue_agents": ", ".join(blue_agents),
                    "bystanders": ", ".join(bystanders),
                    "assassin": ", ".join(assassin),
                    "clue_history": "Previous clues will be shown here",  # TODO: Add actual history
                },
            )

            # Call AI model with metadata tracking
            response, metadata = self.adapter.call_model_with_metadata(self.model_name, prompt)

            # Parse response for clue and number
            logger.debug(f"Raw AI response: {response}")
            clue, number = self._parse_spymaster_response(response)
            
            # Check if we got UNKNOWN and should retry
            if clue == "UNKNOWN" and not is_retry:
                logger.warning(f"Spymaster returned UNKNOWN clue, retrying once...")
                return self._get_spymaster_move_with_retry(board_state, prompt_file, is_retry=True)
            
            # Log AI call metadata (we'll need game context passed from caller)
            # For now, store metadata for potential logging at game level
            self._last_call_metadata = metadata
            self._last_call_metadata["call_type"] = "spymaster"
            self._last_call_metadata["is_retry"] = is_retry
            self._last_call_metadata["turn_result"] = {
                "clue": clue,
                "clue_number": number if isinstance(number, (int, str)) else str(number)
            }

            logger.info(
                f"AI Spymaster ({self.model_name}) gave clue: '{clue}' ({number})" + 
                (" (retry)" if is_retry else "")
            )
            return clue, number

        except Exception as e:
            logger.error(f"Error in AI spymaster move: {e}")
            # If this was not a retry, try once more
            if not is_retry:
                logger.warning(f"Spymaster API call failed, retrying once...")
                return self._get_spymaster_move_with_retry(board_state, prompt_file, is_retry=True)
            # Fallback after retry
            return "ERROR", 1

    def get_referee_validation(
        self, clue: str, number: int|str, team: str, board_state: Dict, prompt_file: str
    ) -> Tuple[bool, str]:
        """Get referee validation of a clue. Returns (is_valid, reasoning)."""
        return self._get_referee_validation_with_retry(clue, number, team, board_state, prompt_file, is_retry=False)

    def _get_referee_validation_with_retry(
        self, clue: str, number: int|str, team: str, board_state: Dict, prompt_file: str, is_retry: bool
    ) -> Tuple[bool, str]:
        """Internal method to get referee validation with retry tracking."""
        try:
            # Get team's agents
            team_agents = [
                word for word, identity in board_state["identities"].items()
                if identity == f"{team}_agent"
            ]
            
            # Load and format prompt
            prompt = self.prompt_manager.load_prompt(
                prompt_file,
                {
                    "clue": clue,
                    "number": number,
                    "team": team,
                    "board": ", ".join(board_state["board"]),
                    "team_agents": ", ".join(team_agents),
                },
            )

            # Call AI model with metadata tracking
            response, metadata = self.adapter.call_model_with_metadata(self.model_name, prompt)

            # Parse response for validation
            is_valid, reasoning = self._parse_referee_response(response)
            
            # Store metadata for logging at game level
            self._last_call_metadata = metadata
            self._last_call_metadata["call_type"] = "referee"
            self._last_call_metadata["is_retry"] = is_retry
            self._last_call_metadata["turn_result"] = {
                "referee_result": "valid" if is_valid else "invalid",
                "referee_reasoning": reasoning
            }

            # Log with full context for debugging if reasoning is generic
            if not is_valid and reasoning in ["Rule violation detected", "Clue approved"]:
                logger.info(
                    f"AI Referee ({self.model_name}) validation: {'VALID' if is_valid else 'INVALID'} - {reasoning} | Full response: {response[:200]}..." +
                    (" (retry)" if is_retry else "")
                )
            else:
                logger.info(
                    f"AI Referee ({self.model_name}) validation: {'VALID' if is_valid else 'INVALID'} - {reasoning}" +
                    (" (retry)" if is_retry else "")
                )
            
            # If invalid, write full prompt+response to logs/referee/
            if not is_valid:
                self._log_referee_foul(clue, number, team, prompt, response, reasoning)
            
            return is_valid, reasoning

        except Exception as e:
            logger.error(f"Error in AI referee validation: {e}")
            # If this was not a retry, try once more
            if not is_retry:
                logger.warning(f"Referee API call failed, retrying once...")
                return self._get_referee_validation_with_retry(clue, number, team, board_state, prompt_file, is_retry=True)
            # Fallback after retry: allow clue but log the error
            return True, f"Referee error - allowing clue: {e}"

    def _log_referee_foul(self, clue: str, number: int|str, team: str, prompt: str, response: str, reasoning: str):
        """Log referee foul details to logs/referee/ directory."""
        try:
            # Create logs/referee directory if it doesn't exist
            referee_log_dir = "logs/referee"
            os.makedirs(referee_log_dir, exist_ok=True)
            
            # Generate filename with timestamp (single file for all fouls)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fouls_{timestamp}.log"
            filepath = os.path.join(referee_log_dir, filename)
            
            # Append foul details (create file if it doesn't exist)
            with open(filepath, 'a') as f:
                f.write(f"=== {team.upper()} TEAM ===\n")
                f.write(f"=== REFEREE FOUL ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Team: {team}\n")
                f.write(f"Clue: {clue}\n")
                f.write(f"Number: {number}\n")
                f.write(f"Foul Reason: {reasoning}\n")
                if reasoning in ["Rule violation detected", "Clue approved"]:
                    f.write(f"NOTE: Generic reasoning detected - check full response below\n")
                f.write(f"Referee Model: {self.model_name}\n\n")
                f.write(f"=== FULL PROMPT ===\n")
                f.write(f"{prompt}\n\n")
                f.write(f"=== REFEREE RESPONSE ===\n")
                f.write(f"{response}\n\n")
                f.write("="*80 + "\n\n")
                
            logger.info(f"Referee foul logged to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to log referee foul: {e}")

    def get_operative_moves(
        self, board_state: Dict, clue: str, number: int|str, prompt_file: str
    ) -> List[str]:
        """Get guesses from AI operative."""
        return self._get_operative_moves_with_retry(board_state, clue, number, prompt_file, is_retry=False)

    def _get_operative_moves_with_retry(
        self, board_state: Dict, clue: str, number: int|str, prompt_file: str, is_retry: bool
    ) -> List[str]:
        """Internal method to get operative moves with retry tracking."""
        try:
            # Load and format prompt
            # Filter board to only show available (unrevealed) words
            available_words = [
                word for word in board_state["board"] 
                if not board_state["revealed"].get(word, False)
            ]
            
            # Format available words as a simple list
            available_words_formatted = ", ".join(available_words)
            
            prompt = self.prompt_manager.load_prompt(
                prompt_file,
                {
                    "BOARD": self._format_board_for_operative(board_state),
                    "AVAILABLE_WORDS": available_words_formatted,
                    "CLUE_HISTORY": board_state.get("clue_history", "None (game just started)"),
                    "CLUE": clue,
                    "NUMBER": number,
                    "TEAM": board_state["current_team"],
                },
            )

            # Call AI model with metadata tracking
            response, metadata = self.adapter.call_model_with_metadata(self.model_name, prompt)

            # Parse response for guesses
            guesses = self._parse_operative_response(response, board_state, number)
            
            # Check if we got empty guesses due to parsing failure and should retry
            if not guesses and not is_retry:
                logger.warning(f"Operative returned no guesses, retrying once...")
                return self._get_operative_moves_with_retry(board_state, clue, number, prompt_file, is_retry=True)
            
            # Store metadata for logging at game level
            self._last_call_metadata = metadata
            self._last_call_metadata["call_type"] = "operative"
            self._last_call_metadata["is_retry"] = is_retry
            self._last_call_metadata["turn_result"] = {
                "total_guesses": len(guesses),
                "guesses": guesses
            }

            logger.info(f"AI Operative ({self.model_name}) guesses: {guesses}" + 
                       (" (retry)" if is_retry else ""))
            return guesses

        except Exception as e:
            logger.error(f"Error in AI operative move: {e}")
            # If this was not a retry, try once more
            if not is_retry:
                logger.warning(f"Operative API call failed, retrying once...")
                return self._get_operative_moves_with_retry(board_state, clue, number, prompt_file, is_retry=True)
            # Fallback after retry
            available = [
                word
                for word in board_state["board"]
                if not board_state["revealed"][word]
            ]
            return available[:1] if available else []

    def _parse_spymaster_response(self, response: str) -> Tuple[str, int|str]:
        """Parse AI response for spymaster clue and number."""
        lines = response.strip().split("\n")

        # Look for clue and number patterns
        clue = "UNKNOWN"
        number: int|str = 1

        for line in lines:
            line = line.strip()
            # Handle both plain and markdown formatted responses
            if line.startswith("CLUE:") or line.startswith("**CLUE:**"):
                clue = line.replace("**CLUE:**", "").replace("CLUE:", "").strip().strip("\"'")
            elif line.startswith("NUMBER:") or line.startswith("**NUMBER:**"):
                number_str = line.replace("**NUMBER:**", "").replace("NUMBER:", "").strip().lower()
                if number_str == "unlimited":
                    number = "unlimited"
                else:
                    try:
                        number = int(number_str)
                    except ValueError:
                        number = 1
            elif ":" in line and len(line.split(":")) == 2:
                # Try to parse "clue: number" format
                parts = line.split(":")
                number_str = parts[1].strip().lower()
                if number_str == "unlimited":
                    clue = parts[0].strip().strip("\"'")
                    number = "unlimited"
                elif number_str.isdigit():
                    clue = parts[0].strip().strip("\"'")
                    number = int(number_str)

        # Ensure valid number (allow 0 and unlimited)
        if isinstance(number, int) and number < 0:
            number = 1

        return clue, number

    def _parse_referee_response(self, response: str) -> Tuple[bool, str]:
        """Parse AI response for referee validation."""
        lines = response.strip().split("\n")
        
        is_valid = True  # Default to valid (allow clue unless clearly invalid)
        reasoning = "Clue approved"
        
        # First pass: look for VALID/INVALID
        found_verdict = False
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("VALID"):
                is_valid = True
                found_verdict = True
                # Look for reasoning on same line
                if ":" in line:
                    reasoning = line.split(":", 1)[1].strip()
                else:
                    reasoning = "Clue follows game rules"
                break
            elif line.startswith("INVALID"):
                is_valid = False
                found_verdict = True
                # Look for reasoning on same line
                if ":" in line:
                    reasoning = line.split(":", 1)[1].strip()
                else:
                    # Look for "Violation:" on subsequent lines
                    reasoning = "Rule violation detected"
                    for next_line in lines[i+1:]:
                        next_line = next_line.strip()
                        if next_line.startswith("Violation:"):
                            reasoning = next_line.replace("Violation:", "").strip()
                            break
                        elif next_line.startswith("Reasoning:"):
                            reasoning = next_line.replace("Reasoning:", "").strip()
                            break
                        elif next_line and not next_line.startswith("#") and not next_line.startswith("**"):
                            # Any other non-empty, non-header line might be the reasoning
                            reasoning = next_line
                            break
                break
        
        # Second pass: look for standalone violation lines if no verdict found
        if not found_verdict:
            for line in lines:
                line = line.strip()
                if line.startswith("Violation:"):
                    is_valid = False
                    reasoning = line.replace("Violation:", "").strip()
                    break
                elif line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
        
        # If no clear reasoning found and clue is invalid, try to extract from full response
        if not is_valid and reasoning == "Rule violation detected":
            # Look for any line that mentions specific violations
            for line in lines:
                line = line.strip().lower()
                if any(keyword in line for keyword in ['multiple words', 'exact match', 'variant', 'letter count', 'position', 'board position']):
                    reasoning = line.title()
                    break
        
        return is_valid, reasoning

    def _format_board_for_operative(self, board_state: Dict) -> str:
        """Format the board for operative display with revealed status."""
        board = board_state["board"]
        revealed = board_state["revealed"]
        
        # Create a 5x5 grid display
        lines = []
        for row in range(5):
            row_items = []
            for col in range(5):
                idx = row * 5 + col
                word = board[idx]
                
                # Mark revealed words with brackets
                if revealed.get(word, False):
                    display_word = f"[{word}]"
                else:
                    display_word = word
                
                row_items.append(f"{display_word:>12}")
            
            lines.append(" |".join(row_items))
        
        return "\n".join(lines)

    def _parse_operative_response(
        self, response: str, board_state: Dict, max_number: int|str
    ) -> List[str]:
        """Parse AI response for operative guesses."""
        available_words = set(
            word for word in board_state["board"] if not board_state["revealed"].get(word, False)
        )
        guesses = []

        # Split response into lines and look for words
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines and obvious non-guess lines
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            # Look for words in the line
            words = line.replace(",", " ").replace(";", " ").split()
            for word in words:
                clean_word = word.strip(".,;:\"'()[]{}").upper()

                # Check if this word is an available word
                for available_word in available_words:
                    if clean_word == available_word.upper():
                        if available_word not in guesses:
                            guesses.append(available_word)
                            # Handle different clue types
                            if max_number == "unlimited" or max_number == 0:
                                # Continue collecting guesses for unlimited/zero clues
                                continue
                            elif isinstance(max_number, int) and len(guesses) >= max_number + 1:  # Plus-one rule
                                return guesses

        # If no valid guesses found, return first available word
        if not guesses and available_words:
            guesses = [next(iter(available_words))]

        # Apply limits based on clue type
        if max_number == "unlimited" or max_number == 0:
            return guesses  # No limit for unlimited/zero clues
        elif isinstance(max_number, int):
            return guesses[: max_number + 1]  # Enforce plus-one limit
        else:
            return guesses  # Fallback
