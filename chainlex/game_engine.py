"""Shared game engine for ChainLex-1.

This module provides the core game logic that is shared between:
- The full game (game.py) for running head-to-head matches
- The optimizer (optimization/) for training prompts

By centralizing the scoring and turn logic here, we ensure the optimizer
learns from the exact same rules as the production game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class BoardState:
    """Immutable representation of a ChainLex-1 board."""
    board: List[str]  # All 16 words
    friendly_words: Set[str]  # 8 target words
    bystanders: Set[str]  # 7 neutral words
    assassin: str  # 1 assassin word
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BoardState":
        """Create BoardState from a dictionary (e.g., puzzle loader output)."""
        return cls(
            board=data["board"],
            friendly_words=set(w.upper() for w in data["friendly"]),
            bystanders=set(w.upper() for w in data["bystanders"]),
            assassin=data["assassin"].upper(),
        )
    
    @classmethod
    def from_strings(
        cls,
        board: str,
        friendly_words: str,
        bystanders: str,
        assassin: str,
    ) -> "BoardState":
        """Create BoardState from comma-separated strings (DSPy example format)."""
        return cls(
            board=[w.strip().upper() for w in board.split(",")],
            friendly_words=set(w.strip().upper() for w in friendly_words.split(",")),
            bystanders=set(w.strip().upper() for w in bystanders.split(",")),
            assassin=assassin.strip().upper(),
        )


@dataclass
class TurnResult:
    """Result of playing a single turn."""
    score: int
    correct_count: int
    guesses_made: List[str]
    end_reason: str  # "complete", "bystander", "assassin"
    hit_assassin: bool
    hit_bystander: bool
    
    @property
    def is_catastrophic(self) -> bool:
        """True if the turn ended with an assassin hit."""
        return self.hit_assassin


class GameEngine:
    """Core game engine for ChainLex-1.
    
    Provides the scoring logic used by both the full game and the optimizer.
    This is the single source of truth for game rules.
    """
    
    # Scoring constants (GAMEPLAY mode - production rules)
    BYSTANDER_PENALTY = -1
    ASSASSIN_PENALTY = -1000  # Instant loss
    
    @classmethod
    def score_guesses(
        cls,
        guesses: List[str],
        board_state: BoardState,
        max_guesses: Optional[int] = None,
    ) -> TurnResult:
        """Score a list of guesses according to ChainLex-1 rules.
        
        This is the ONLY place where scoring logic should be implemented.
        Both the game and optimizer use this method.
        
        Args:
            guesses: Ordered list of guesses (most confident first)
            board_state: The board configuration
            max_guesses: Maximum guesses allowed (usually clue number)
        
        Returns:
            TurnResult with score and metadata
        """
        score = 0
        correct_count = 0
        guesses_made = []
        end_reason = "complete"
        hit_assassin = False
        hit_bystander = False
        
        # Limit guesses if max specified
        if max_guesses is not None:
            guesses = guesses[:max_guesses]
        
        for guess in guesses:
            guess_upper = guess.upper()
            guesses_made.append(guess_upper)
            
            if guess_upper == board_state.assassin:
                # Assassin - instant loss
                score += cls.ASSASSIN_PENALTY
                end_reason = "assassin"
                hit_assassin = True
                break
            elif guess_upper in board_state.bystanders:
                # Bystander - penalty and stop
                score += cls.BYSTANDER_PENALTY
                end_reason = "bystander"
                hit_bystander = True
                break
            elif guess_upper in board_state.friendly_words:
                # Correct - triangular scoring (1st=1, 2nd=2, 3rd=3, etc.)
                correct_count += 1
                score += correct_count
            # else: word not on board, skip (shouldn't happen with good parsing)
        
        return TurnResult(
            score=score,
            correct_count=correct_count,
            guesses_made=guesses_made,
            end_reason=end_reason,
            hit_assassin=hit_assassin,
            hit_bystander=hit_bystander,
        )
    
    @classmethod
    def max_possible_score(cls, num_friendly: int = 8) -> int:
        """Calculate maximum possible score (all friendly words correct).
        
        Triangular number: 1 + 2 + 3 + ... + n = n(n+1)/2
        """
        return num_friendly * (num_friendly + 1) // 2
    
    @classmethod
    def parse_guesses_from_response(
        cls,
        response: str,
        available_words: Set[str],
        max_guesses: int,
    ) -> List[str]:
        """Parse guesses from an LLM response.
        
        This is the canonical parsing logic used by both game and optimizer.
        Looks for a "Guesses:" section first, then falls back to scanning.
        
        Args:
            response: Raw LLM response text
            available_words: Set of valid words that can be guessed
            max_guesses: Maximum number of guesses to extract
        
        Returns:
            List of parsed guesses (uppercase)
        """
        available_upper = {w.upper() for w in available_words}
        guesses = []
        
        lines = response.strip().split("\n")
        
        # First pass: look for explicit "Guesses:" section
        in_guesses_section = False
        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Detect start of guesses section
            if line_upper.startswith("GUESSES") or line_upper.startswith("### GUESSES") or line_upper.startswith("## GUESSES"):
                in_guesses_section = True
                # Check if guesses are on the same line
                if ":" in line_stripped:
                    guess_part = line_stripped.split(":", 1)[1].strip()
                    for word in guess_part.replace(",", " ").split():
                        clean_word = word.strip(".,;:\"'()[]{}→*`").upper()
                        if clean_word in available_upper and clean_word not in guesses:
                            guesses.append(clean_word)
                            if len(guesses) >= max_guesses:
                                return guesses
                continue
            
            # If in guesses section, extract words
            if in_guesses_section:
                if line_stripped.startswith("#") or line_stripped.startswith("---"):
                    break
                if not line_stripped and guesses:
                    break
                
                for word in line_stripped.replace(",", " ").split():
                    clean_word = word.strip(".,;:\"'()[]{}→*`").upper()
                    if clean_word in available_upper and clean_word not in guesses:
                        guesses.append(clean_word)
                        if len(guesses) >= max_guesses:
                            return guesses
        
        if guesses:
            return guesses[:max_guesses]
        
        # Fallback: scan for board words in short lines
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue
            if "reject" in line_stripped.lower() or "trap" in line_stripped.lower():
                continue
            
            words = line_stripped.replace(",", " ").replace(";", " ").split()
            if len(words) <= 5:
                for word in words:
                    clean_word = word.strip(".,;:\"'()[]{}→*`").upper()
                    if clean_word in available_upper and clean_word not in guesses:
                        guesses.append(clean_word)
                        if len(guesses) >= max_guesses:
                            return guesses
        
        # If still nothing, return first available word
        if not guesses and available_upper:
            guesses = [next(iter(available_upper))]
        
        return guesses[:max_guesses]
    
    @classmethod
    def parse_clue_from_response(cls, response: str) -> Tuple[str, int]:
        """Parse clue and number from an LLM response.
        
        This is the canonical parsing logic used by both game and optimizer.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            Tuple of (clue, number)
        """
        lines = response.strip().split("\n")
        clue = "UNKNOWN"
        number = 1
        
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            
            line_upper = line.upper()
            
            # Parse CLUE
            if line_upper.startswith("CLUE") or line_upper.startswith("**CLUE") or line_upper.startswith("`CLUE"):
                if ":" in line:
                    clue_part = line.split(":", 1)[1].strip()
                    clue = clue_part.strip("*\"'` ")
                    if clue:
                        continue
            
            # Parse NUMBER
            if line_upper.startswith("NUMBER") or line_upper.startswith("**NUMBER"):
                if ":" in line:
                    number_part = line.split(":", 1)[1].strip()
                    number_str = ""
                    for char in number_part:
                        if char.isdigit():
                            number_str += char
                        elif number_str:
                            break
                    if number_str:
                        try:
                            number = int(number_str)
                        except ValueError:
                            number = 1
                    continue
            
            # Fallback: "clue: number" format
            if ":" in line and len(line.split(":")) == 2:
                parts = line.split(":")
                if parts[1].strip().isdigit():
                    clue = parts[0].strip().strip("\"'*`")
                    number = int(parts[1].strip())
        
        # Validate
        number = max(1, min(8, number))
        
        return clue, number

