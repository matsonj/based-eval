"""Scoring metrics for ChainLex-1 DSPy optimization.

Implements the same scoring as the game:
- Correct guesses: triangular scoring (1+2+3+...)
- Bystander: -5 points
- Assassin: -28 points
"""

from typing import List, Set


# Scoring constants (match game.py)
BYSTANDER_PENALTY = -5
ASSASSIN_PENALTY = -28


def score_guesses(
    guesses: List[str],
    friendly_words: Set[str],
    bystanders: Set[str],
    assassin: str,
) -> int:
    """Score a list of guesses according to ChainLex-1 rules.
    
    Args:
        guesses: Ordered list of guesses (most confident first)
        friendly_words: Set of target words
        bystanders: Set of neutral words
        assassin: The assassin word
    
    Returns:
        Final score (can be negative)
    """
    score = 0
    correct_count = 0
    
    for guess in guesses:
        guess_upper = guess.upper()
        
        if guess_upper == assassin.upper():
            # Assassin - catastrophic
            score += ASSASSIN_PENALTY
            break
        elif guess_upper in {w.upper() for w in bystanders}:
            # Bystander - penalty and stop
            score += BYSTANDER_PENALTY
            break
        elif guess_upper in {w.upper() for w in friendly_words}:
            # Correct - triangular scoring
            correct_count += 1
            score += correct_count
        # else: word not on board, skip (shouldn't happen)
    
    return score


def chainlex_metric(example, prediction, trace=None) -> float:
    """DSPy metric function for ChainLex-1 optimization.
    
    Evaluates a prediction by scoring the guesses against the known board state.
    
    Args:
        example: DSPy example with board state (friendly_words, bystanders, assassin)
        prediction: DSPy prediction with guesses list
        trace: Optional trace (unused)
    
    Returns:
        Score as a float (higher is better)
    """
    # Parse friendly words, bystanders, assassin from example
    friendly_words = set(
        word.strip().upper() 
        for word in str(example.friendly_words).split(',')
    )
    bystanders = set(
        word.strip().upper() 
        for word in str(example.bystanders).split(',')
    )
    assassin = str(example.assassin).strip().upper()
    
    # Get guesses from prediction
    guesses = prediction.guesses if hasattr(prediction, 'guesses') else []
    
    # Handle string guesses (comma-separated)
    if isinstance(guesses, str):
        guesses = [g.strip() for g in guesses.split(',')]
    
    # Score the guesses
    score = score_guesses(guesses, friendly_words, bystanders, assassin)
    
    return float(score)


def max_possible_score(num_friendly: int = 8) -> int:
    """Calculate maximum possible score (all friendly words correct).
    
    Triangular number: 1 + 2 + 3 + ... + n = n(n+1)/2
    """
    return num_friendly * (num_friendly + 1) // 2


def normalized_metric(example, prediction, trace=None) -> float:
    """Normalized version of the metric (0-1 scale for well-behaved optimization).
    
    Maps score to [0, 1] range where:
    - 0 = assassin hit (-28)
    - 0.1 = bystander hit with no correct (-5)
    - 0.5 = score of 0
    - 1.0 = perfect score (36)
    
    This helps optimizers that expect metrics in [0, 1].
    """
    raw_score = chainlex_metric(example, prediction, trace)
    max_score = max_possible_score()
    
    # Handle catastrophic outcomes
    if raw_score <= ASSASSIN_PENALTY:
        return 0.0
    
    # Normalize: map [-5, 36] to [0.1, 1.0]
    # Using linear interpolation
    min_normal_score = BYSTANDER_PENALTY  # -5
    
    if raw_score < 0:
        # Negative scores (bystander hit): map to [0.1, 0.5]
        # -5 -> 0.1, 0 -> 0.5
        return 0.1 + 0.4 * (raw_score - min_normal_score) / (-min_normal_score)
    else:
        # Positive scores: map to [0.5, 1.0]
        # 0 -> 0.5, 36 -> 1.0
        return 0.5 + 0.5 * (raw_score / max_score)

