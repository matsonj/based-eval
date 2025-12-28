"""Scoring metrics for ChainLex-1 DSPy optimization.

Uses the GameEngine from game_engine.py to ensure consistent scoring
between the optimizer and the production game.
"""

from typing import List, Set

# Import the shared game engine - single source of truth
from chainlex.game_engine import GameEngine, BoardState


# Re-export constants from GameEngine for backward compatibility
BYSTANDER_PENALTY = GameEngine.BYSTANDER_PENALTY
ASSASSIN_PENALTY = GameEngine.ASSASSIN_PENALTY


def score_guesses(
    guesses: List[str],
    friendly_words: Set[str],
    bystanders: Set[str],
    assassin: str,
) -> int:
    """Score a list of guesses according to ChainLex-1 rules.
    
    This is a thin wrapper around GameEngine.score_guesses for backward compatibility.
    
    Args:
        guesses: Ordered list of guesses (most confident first)
        friendly_words: Set of target words
        bystanders: Set of neutral words
        assassin: The assassin word
    
    Returns:
        Final score (can be negative)
    """
    board_state = BoardState(
        board=[],  # Not used for scoring
        friendly_words={w.upper() for w in friendly_words},
        bystanders={w.upper() for w in bystanders},
        assassin=assassin.upper(),
    )
    
    result = GameEngine.score_guesses(guesses, board_state)
    return result.score


def chainlex_metric(example, prediction, trace=None) -> float:
    """DSPy metric function for ChainLex-1 optimization.
    
    Uses GameEngine.score_guesses for consistent scoring with production.
    
    Args:
        example: DSPy example with board state (friendly_words, bystanders, assassin)
        prediction: DSPy prediction with guesses list
        trace: Optional trace (unused)
    
    Returns:
        Score as a float (higher is better)
    """
    # Create BoardState from DSPy example
    board_state = BoardState.from_strings(
        board=str(example.board) if hasattr(example, 'board') else "",
        friendly_words=str(example.friendly_words),
        bystanders=str(example.bystanders),
        assassin=str(example.assassin),
    )
    
    # Get guesses from prediction
    guesses = prediction.guesses if hasattr(prediction, 'guesses') else []
    
    # Handle string guesses (comma-separated)
    if isinstance(guesses, str):
        guesses = [g.strip() for g in guesses.split(',')]
    
    # Score using the game engine
    result = GameEngine.score_guesses(guesses, board_state)
    
    return float(result.score)


def max_possible_score(num_friendly: int = 8) -> int:
    """Calculate maximum possible score (all friendly words correct)."""
    return GameEngine.max_possible_score(num_friendly)


def normalized_metric(example, prediction, trace=None) -> float:
    """Normalized version of the metric (0-1 scale for well-behaved optimization).
    
    Maps score to [0, 1] range where:
    - 0 = assassin hit (instant loss)
    - 0.1 = bystander hit with no correct (-1)
    - 0.5 = score of 0
    - 1.0 = perfect score (36)
    
    This helps optimizers that expect metrics in [0, 1].
    """
    raw_score = chainlex_metric(example, prediction, trace)
    max_score = max_possible_score()
    
    # Handle catastrophic outcomes (assassin = instant loss)
    if raw_score <= -100:  # Assassin threshold
        return 0.0
    
    # Normalize: map [BYSTANDER_PENALTY, 36] to [0.1, 1.0]
    min_normal_score = BYSTANDER_PENALTY  # -1 in gameplay mode
    
    if raw_score < 0:
        # Negative scores (bystander hit): map to [0.1, 0.5]
        if min_normal_score == 0:
            return 0.5
        return 0.1 + 0.4 * (raw_score - min_normal_score) / (-min_normal_score)
    else:
        # Positive scores: map to [0.5, 1.0]
        # 0 -> 0.5, 36 -> 1.0
        return 0.5 + 0.5 * (raw_score / max_score)
