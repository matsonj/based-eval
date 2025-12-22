"""Scoring metrics for DSPy optimization of Codenames prompts."""

from dataclasses import dataclass
from typing import List, Optional, Any

from codenames.optimization.data_extractor import ClueExample, GuessOutcome, GuessResult


# Scoring weights (matching user requirements)
SCORE_CORRECT = 1.0
SCORE_BYSTANDER = -0.5
SCORE_ENEMY = -1.0
SCORE_ASSASSIN = -999.0


def guess_score(outcome: GuessOutcome) -> float:
    """Score for a single guess outcome."""
    scores = {
        GuessOutcome.CORRECT: SCORE_CORRECT,
        GuessOutcome.BYSTANDER: SCORE_BYSTANDER,
        GuessOutcome.ENEMY: SCORE_ENEMY,
        GuessOutcome.ASSASSIN: SCORE_ASSASSIN,
    }
    return scores[outcome]


def score_guess_sequence(guesses: List[GuessResult]) -> float:
    """Score a sequence of guesses.
    
    Returns:
        Total score: sum of individual guess scores
        +1 per correct, -0.5 per bystander, -1 per enemy, -999 per assassin
    """
    return sum(guess_score(g.outcome) for g in guesses)


@dataclass
class CodenamesTotalScore:
    """Comprehensive scoring result for DSPy optimization."""
    
    # Raw counts
    correct_count: int
    bystander_count: int
    enemy_count: int
    assassin_count: int
    
    # Clue metadata
    clue_number: int  # How many words the spymaster intended
    total_guesses: int  # How many guesses were made
    
    # Computed scores
    raw_score: float  # Sum of guess scores
    accuracy: float  # correct_count / clue_number (capped at 1.0)
    efficiency: float  # correct_count / total_guesses (if any guesses)
    
    @property 
    def normalized_score(self) -> float:
        """Score normalized by clue ambition (clue_number).
        
        A clue targeting 3 words that gets 2 correct is worse than
        a clue targeting 2 words that gets 2 correct.
        """
        if self.clue_number == 0:
            return self.raw_score
        return self.raw_score / self.clue_number
    
    @property
    def is_catastrophic(self) -> bool:
        """Whether this resulted in an assassin hit."""
        return self.assassin_count > 0
    
    def __float__(self) -> float:
        """Convert to float for DSPy metric."""
        return self.raw_score


def codenames_score(
    guesses: List[str],
    team_agents: List[str],
    enemy_agents: List[str],
    bystanders: List[str],
    assassin: str,
    clue_number: int = 1,
) -> CodenamesTotalScore:
    """Score a sequence of guesses against the board state.
    
    This is the primary metric function for DSPy optimization.
    
    Args:
        guesses: List of words guessed by operative
        team_agents: Words that are this team's agents
        enemy_agents: Words that are enemy agents
        bystanders: Neutral words
        assassin: The assassin word
        clue_number: How many words the clue was targeting
        
    Returns:
        CodenamesTotalScore with all metrics computed
    """
    correct_count = 0
    bystander_count = 0
    enemy_count = 0
    assassin_count = 0
    
    # Normalize for case-insensitive comparison
    team_agents_upper = {w.upper() for w in team_agents}
    enemy_agents_upper = {w.upper() for w in enemy_agents}
    bystanders_upper = {w.upper() for w in bystanders}
    assassin_upper = assassin.upper()
    
    for guess in guesses:
        guess_upper = guess.upper()
        
        if guess_upper == assassin_upper:
            assassin_count += 1
        elif guess_upper in team_agents_upper:
            correct_count += 1
        elif guess_upper in enemy_agents_upper:
            enemy_count += 1
        elif guess_upper in bystanders_upper:
            bystander_count += 1
        # Unknown words are ignored (shouldn't happen in valid games)
    
    raw_score = (
        correct_count * SCORE_CORRECT +
        bystander_count * SCORE_BYSTANDER +
        enemy_count * SCORE_ENEMY +
        assassin_count * SCORE_ASSASSIN
    )
    
    total_guesses = len(guesses)
    
    # Accuracy: what fraction of intended words were found
    if clue_number > 0:
        accuracy = min(1.0, correct_count / clue_number)
    else:
        accuracy = 1.0 if correct_count > 0 else 0.0
    
    # Efficiency: what fraction of guesses were correct
    if total_guesses > 0:
        efficiency = correct_count / total_guesses
    else:
        efficiency = 0.0
    
    return CodenamesTotalScore(
        correct_count=correct_count,
        bystander_count=bystander_count,
        enemy_count=enemy_count,
        assassin_count=assassin_count,
        clue_number=clue_number,
        total_guesses=total_guesses,
        raw_score=raw_score,
        accuracy=accuracy,
        efficiency=efficiency,
    )


def dspy_metric(example: Any, prediction: Any, trace: Optional[Any] = None) -> float:
    """DSPy-compatible metric function for optimization.
    
    This function will be passed to DSPy optimizers. It takes:
    - example: The input example (contains board state, expected outputs)
    - prediction: The model's prediction (contains clue, guesses)
    - trace: Optional trace for debugging
    
    Returns:
        Float score (higher is better)
    """
    # Extract board state from example
    team_agents = example.team_agents
    enemy_agents = example.enemy_agents
    bystanders = example.bystanders
    assassin = example.assassin
    
    # Extract prediction outputs
    # The prediction object structure depends on our DSPy module design
    guesses = getattr(prediction, 'guesses', [])
    clue_number = getattr(prediction, 'clue_number', 1)
    
    # Handle case where guesses is a string (single guess)
    if isinstance(guesses, str):
        guesses = [guesses]
    
    # Compute score
    score_result = codenames_score(
        guesses=guesses,
        team_agents=team_agents,
        enemy_agents=enemy_agents,
        bystanders=bystanders,
        assassin=assassin,
        clue_number=clue_number,
    )
    
    return score_result.raw_score


def create_dspy_metric(normalize_by_clue_number: bool = False):
    """Factory function to create DSPy metric with configurable behavior.
    
    Args:
        normalize_by_clue_number: If True, divide score by clue_number
            (rewards ambitious clues that succeed)
    """
    def metric_fn(example: Any, prediction: Any, trace: Optional[Any] = None) -> float:
        team_agents = example.team_agents
        enemy_agents = example.enemy_agents
        bystanders = example.bystanders
        assassin = example.assassin
        
        guesses = getattr(prediction, 'guesses', [])
        clue_number = getattr(prediction, 'clue_number', 1)
        
        if isinstance(guesses, str):
            guesses = [guesses]
        
        score_result = codenames_score(
            guesses=guesses,
            team_agents=team_agents,
            enemy_agents=enemy_agents,
            bystanders=bystanders,
            assassin=assassin,
            clue_number=clue_number,
        )
        
        if normalize_by_clue_number:
            return score_result.normalized_score
        return score_result.raw_score
    
    return metric_fn

