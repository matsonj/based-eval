"""ChainLex-1: A single-player word association game for LLM evaluation.

ChainLex-1 is a simplified variant of Codenames designed for efficient LLM evaluation.
Key differences from Codenames:
- Single player (same LLM gives clues and guesses)
- One round only
- Smaller board (16 words: 8 friendly, 7 bystanders, 1 assassin)
- Triangular scoring: nth correct guess = n points
- Bystander: -5 points (ends turn)
- Assassin: -28 points (ends game)
"""

from chainlex.game import ChainLexGame

__all__ = ["ChainLexGame"]

