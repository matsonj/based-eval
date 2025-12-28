# ChainLex-1 - Core Game Rules

## Game Overview
ChainLex-1 is a single-player word association game designed for LLM evaluation. You see a grid of 16 words and must identify the 8 "friendly" words while avoiding bystanders and the assassin. Your goal is to maximize your score in a single turn.

## Board Setup
- **16 words total** arranged in a 4x4 grid
- **8 Friendly Words** - These are YOUR target words
- **7 Bystanders** - Neutral words (penalty if guessed)
- **1 Assassin** - Catastrophic if guessed

## Scoring System
Your score is calculated as follows:
- **Correct Guesses**: Triangular scoring - the nth correct guess earns n points
  - 1st correct = 1 point
  - 2nd correct = 2 points
  - 3rd correct = 3 points
  - And so on...
  - All 8 correct = 1+2+3+4+5+6+7+8 = 36 points (maximum)
- **Bystander**: -1 point (ends your turn immediately)
- **Assassin**: Instant loss (-1000 points)

## Game Flow
1. **Clue Phase**: You see all 16 words WITH their identities (friendly, bystander, assassin). You provide a single-word clue and a number.
2. **Guess Phase**: You see the same 16 words WITHOUT identities. Using only your clue, you make ordered guesses.
3. **Resolution**: Guesses are processed in order until you hit a non-friendly word or run out of guesses.

## Strategy
- **Risk vs Reward**: Going for more words means higher potential score but also higher risk
- **Safe Clues**: Targeting 2-3 words is safer but limits max score
- **Aggressive Clues**: Targeting 5+ words could yield huge points but one mistake ends it all
- **Best Outcome**: Find all 8 friendly words without hitting any bystanders or the assassin

## Clue Validation Rules

### Valid Clues Must:
1. **Be a single word** (compound words and proper nouns allowed)
2. **NOT be a word currently on the board** (exact match forbidden)
3. **NOT be a grammatical variant of a board word** (plural, past tense, etc.)
4. **NOT reference the number of letters** in target words
5. **NOT reference positions** on the board (row, column, location)

### Single Word Exceptions
1. Compound words are allowed when describing specific ideas (e.g., "Greenhouse")
2. Proper names are always valid if they follow other rules (e.g., "Shakespeare", "New York")
3. Common acronyms and abbreviations are allowed (e.g., "CIA", "NASA", "SQL")

