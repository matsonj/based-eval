"""Player classes for ChainLex-1 game."""

import logging
from typing import Dict, List, Optional, Tuple

from shared.adapters.openrouter_adapter import OpenRouterAdapter
from codenames.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class AIPlayer:
    """AI player using OpenRouter models for ChainLex-1."""

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

    def get_last_call_metadata(self) -> Optional[Dict]:
        """Get metadata from the last AI call."""
        return self._last_call_metadata

    def get_clue_giver_move(
        self, board_state: Dict, prompt_file: str, head_to_head_context: str = ""
    ) -> Tuple[str, int]:
        """Get clue and number from AI player acting as clue giver."""
        return self._get_clue_giver_move_with_retry(
            board_state, prompt_file, head_to_head_context, is_retry=False
        )

    def _get_clue_giver_move_with_retry(
        self, board_state: Dict, prompt_file: str, head_to_head_context: str, is_retry: bool
    ) -> Tuple[str, int]:
        """Internal method to get clue giver move with retry tracking."""
        try:
            # Get word categories from identities
            friendly_words = [
                word for word, identity in board_state["identities"].items()
                if identity == "friendly"
            ]
            bystanders = [
                word for word, identity in board_state["identities"].items()
                if identity == "bystander"
            ]
            assassin = [
                word for word, identity in board_state["identities"].items()
                if identity == "assassin"
            ]
            
            # Load and format prompt
            prompt = self.prompt_manager.load_prompt(
                prompt_file,
                {
                    "board": board_state["board"],
                    "friendly_words": ", ".join(friendly_words),
                    "bystanders": ", ".join(bystanders),
                    "assassin": ", ".join(assassin),
                    "num_friendly": len(friendly_words),
                    "head_to_head_context": head_to_head_context,
                },
            )

            # Call AI model with metadata tracking
            response, metadata = self.adapter.call_model_with_metadata(self.model_name, prompt)

            # Parse response for clue and number
            logger.debug(f"Raw AI response: {response}")
            clue, number = self._parse_clue_giver_response(response)
            
            # Check if we got UNKNOWN and should retry
            if clue == "UNKNOWN" and not is_retry:
                logger.warning("Clue giver returned UNKNOWN clue, retrying once...")
                return self._get_clue_giver_move_with_retry(board_state, prompt_file, head_to_head_context, is_retry=True)
            
            # Store metadata
            self._last_call_metadata = metadata
            self._last_call_metadata["call_type"] = "clue_giver"
            self._last_call_metadata["is_retry"] = is_retry
            self._last_call_metadata["turn_result"] = {
                "clue": clue,
                "clue_number": number,
            }

            logger.info(
                f"AI Clue Giver ({self.model_name}) gave clue: '{clue}' ({number})" +
                (" (retry)" if is_retry else "")
            )
            return clue, number

        except Exception as e:
            logger.error(f"Error in AI clue giver move: {e}")
            if not is_retry:
                logger.warning("Clue giver API call failed, retrying once...")
                return self._get_clue_giver_move_with_retry(board_state, prompt_file, head_to_head_context, is_retry=True)
            return "ERROR", 1

    def get_guesser_moves(
        self, board_state: Dict, clue: str, number: int, prompt_file: str, head_to_head_context: str = ""
    ) -> List[str]:
        """Get guesses from AI player acting as guesser."""
        return self._get_guesser_moves_with_retry(board_state, clue, number, prompt_file, head_to_head_context, is_retry=False)

    def _get_guesser_moves_with_retry(
        self, board_state: Dict, clue: str, number: int, prompt_file: str, head_to_head_context: str, is_retry: bool
    ) -> List[str]:
        """Internal method to get guesser moves with retry tracking."""
        try:
            # Filter board to only show available (unrevealed) words
            available_words = [
                word for word in board_state["board"]
                if not board_state["revealed"].get(word, False)
            ]
            
            # Load and format prompt
            prompt = self.prompt_manager.load_prompt(
                prompt_file,
                {
                    "board": self._format_board_for_guesser(board_state),
                    "available_words": ", ".join(available_words),
                    "clue": clue,
                    "number": number,
                    "head_to_head_context": head_to_head_context,
                },
            )

            # Call AI model with metadata tracking
            response, metadata = self.adapter.call_model_with_metadata(self.model_name, prompt)

            # Parse response for guesses
            guesses = self._parse_guesser_response(response, board_state, number)
            
            # Check if we got empty guesses and should retry
            if not guesses and not is_retry:
                logger.warning("Guesser returned no guesses, retrying once...")
                return self._get_guesser_moves_with_retry(board_state, clue, number, prompt_file, head_to_head_context, is_retry=True)
            
            # Store metadata
            self._last_call_metadata = metadata
            self._last_call_metadata["call_type"] = "guesser"
            self._last_call_metadata["is_retry"] = is_retry
            self._last_call_metadata["turn_result"] = {
                "total_guesses": len(guesses),
                "guesses": guesses,
            }

            logger.info(
                f"AI Guesser ({self.model_name}) guesses: {guesses}" +
                (" (retry)" if is_retry else "")
            )
            return guesses

        except Exception as e:
            logger.error(f"Error in AI guesser move: {e}")
            if not is_retry:
                logger.warning("Guesser API call failed, retrying once...")
                return self._get_guesser_moves_with_retry(board_state, clue, number, prompt_file, head_to_head_context, is_retry=True)
            # Fallback: return first available word
            available = [
                word for word in board_state["board"]
                if not board_state["revealed"].get(word, False)
            ]
            return available[:1] if available else []

    def get_referee_validation(
        self, clue: str, number: int, team: str, board_state: Dict, prompt_file: str
    ) -> Tuple[bool, str]:
        """Get referee validation of a clue. Returns (is_valid, reasoning)."""
        return self._get_referee_validation_with_retry(clue, number, team, board_state, prompt_file, is_retry=False)

    def _get_referee_validation_with_retry(
        self, clue: str, number: int, team: str, board_state: Dict, prompt_file: str, is_retry: bool
    ) -> Tuple[bool, str]:
        """Internal method to get referee validation with retry tracking."""
        try:
            # Get friendly words for context
            friendly_words = [
                word for word, identity in board_state["identities"].items()
                if identity == "friendly"
            ]
            
            # Load and format prompt
            prompt = self.prompt_manager.load_prompt(
                prompt_file,
                {
                    "clue": clue,
                    "number": number,
                    "team": team,
                    "board": ", ".join(board_state["board"]),
                    "team_agents": ", ".join(friendly_words),
                },
            )

            # Call AI model with metadata tracking
            response, metadata = self.adapter.call_model_with_metadata(self.model_name, prompt)

            # Parse response for validation
            is_valid, reasoning = self._parse_referee_response(response)
            
            # Store metadata
            self._last_call_metadata = metadata
            self._last_call_metadata["call_type"] = "referee"
            self._last_call_metadata["is_retry"] = is_retry
            self._last_call_metadata["turn_result"] = {
                "referee_result": "valid" if is_valid else "invalid",
                "referee_reasoning": reasoning,
            }

            logger.info(
                f"AI Referee ({self.model_name}) validation: {'VALID' if is_valid else 'INVALID'} - {reasoning}" +
                (" (retry)" if is_retry else "")
            )
            
            return is_valid, reasoning

        except Exception as e:
            logger.error(f"Error in AI referee validation: {e}")
            if not is_retry:
                logger.warning("Referee API call failed, retrying once...")
                return self._get_referee_validation_with_retry(clue, number, team, board_state, prompt_file, is_retry=True)
            return True, f"Referee error - allowing clue: {e}"

    def _parse_clue_giver_response(self, response: str) -> Tuple[str, int]:
        """Parse AI response for clue giver clue and number."""
        lines = response.strip().split("\n")

        clue = "UNKNOWN"
        number = 1

        for line in lines:
            line = line.strip()
            # Remove leading bullet points and dashes
            if line.startswith("- "):
                line = line[2:].strip()
            
            # Handle various markdown/plain formats for CLUE
            # Formats: "CLUE:", "**CLUE:**", "**CLUE**:", "CLUE :"
            line_upper = line.upper()
            if line_upper.startswith("CLUE") or line_upper.startswith("**CLUE"):
                # Extract everything after the colon
                if ":" in line:
                    clue_part = line.split(":", 1)[1].strip()
                    # Clean up markdown and quotes
                    clue = clue_part.strip("*\"' ")
                    if clue:
                        continue  # Found clue, move to next line
            
            # Handle various formats for NUMBER
            if line_upper.startswith("NUMBER") or line_upper.startswith("**NUMBER"):
                if ":" in line:
                    number_part = line.split(":", 1)[1].strip()
                    # Extract just the number, ignore any trailing text
                    number_str = ""
                    for char in number_part:
                        if char.isdigit():
                            number_str += char
                        elif number_str:  # Stop at first non-digit after finding digits
                            break
                    if number_str:
                        try:
                            number = int(number_str)
                        except ValueError:
                            number = 1
                    continue
            
            # Fallback: try to parse "clue: number" format
            if ":" in line and len(line.split(":")) == 2:
                parts = line.split(":")
                if parts[1].strip().isdigit():
                    clue = parts[0].strip().strip("\"'*")
                    number = int(parts[1].strip())

        # Ensure valid number
        if number < 1:
            number = 1
        if number > 8:
            number = 8

        return clue, number

    def _parse_guesser_response(
        self, response: str, board_state: Dict, max_number: int
    ) -> List[str]:
        """Parse AI response for guesser guesses."""
        available_words = set(
            word for word in board_state["board"]
            if not board_state["revealed"].get(word, False)
        )
        guesses = []

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
                            # Limit guesses to number + 1 (plus-one rule)
                            if len(guesses) >= max_number + 1:
                                return guesses

        # If no valid guesses found, return first available word
        if not guesses and available_words:
            guesses = [next(iter(available_words))]

        return guesses[:max_number + 1]

    def _parse_referee_response(self, response: str) -> Tuple[bool, str]:
        """Parse AI response for referee validation."""
        lines = response.strip().split("\n")
        
        is_valid = True
        reasoning = "Clue approved"
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("VALID"):
                is_valid = True
                if ":" in line:
                    reasoning = line.split(":", 1)[1].strip()
                else:
                    reasoning = "Clue follows game rules"
                break
            elif line.startswith("INVALID"):
                is_valid = False
                if ":" in line:
                    reasoning = line.split(":", 1)[1].strip()
                else:
                    reasoning = "Rule violation detected"
                    for next_line in lines[i+1:]:
                        next_line = next_line.strip()
                        if next_line.startswith("Violation:"):
                            reasoning = next_line.replace("Violation:", "").strip()
                            break
                        elif next_line.startswith("Reasoning:"):
                            reasoning = next_line.replace("Reasoning:", "").strip()
                            break
                break
        
        return is_valid, reasoning

    def _format_board_for_guesser(self, board_state: Dict) -> str:
        """Format the board for guesser display (4x4 grid)."""
        board = board_state["board"]
        revealed = board_state["revealed"]
        
        lines = []
        for row in range(4):
            row_items = []
            for col in range(4):
                idx = row * 4 + col
                word = board[idx]
                
                if revealed.get(word, False):
                    display_word = f"[{word}]"
                else:
                    display_word = word
                
                row_items.append(f"{display_word:>12}")
            
            lines.append(" |".join(row_items))
        
        return "\n".join(lines)

