You are playing ChainLex-1, a word association game. Guess words based on the clue to maximize your score.

{{HEAD_TO_HEAD_CONTEXT}}

## Scoring
- Each correct guess earns points: 1st=1pt, 2nd=2pt, 3rd=3pt, etc.
- Bystander (neutral word): -5 points and ENDS your turn
- Assassin: -28 points and ENDS your turn

## Your Goal
Identify the friendly words that match the clue. Guess in order of confidence - most confident first.

## Rules
- You can guess up to {{NUMBER}} words
- Guesses are processed in order - wrong guess ENDS your turn
- Order matters: put your most confident guesses first

## Strategy
- Be aggressive if confident - more correct guesses = exponentially more points
- If unsure about later guesses, stop early - a bystander costs -5 points AND ends your turn
- The assassin is catastrophic (-28) - avoid at all costs

## Examples

**Example 1:**
Board: PASTA, PORT, SPY, YACHT, SALARY, CODE, FORTRESS, CHEESE, BRACELET, TRAIL, LION, WOLF, EAGLE, SHARK, TIGER, BEAR
Clue: PREDATOR (6)
Reasoning: Animals that are predators: LION, WOLF, EAGLE, SHARK, TIGER, BEAR are all predators. These 6 words fit perfectly. SPY could be metaphorically predatory but it's risky. Go with the 6 clear animal predators.
→ Guesses: LION, WOLF, SHARK, TIGER, EAGLE, BEAR

**Example 2:**
Board: PIANO, MIRROR, CROWN, DIAMOND, CLOCK, BELL, BOOK, TELESCOPE, COMPASS, ANCHOR, KEY, LAMP, COFFEE, HONEY, SALT, APPLE
Clue: KEYS (4)
Reasoning: Things with keys: PIANO has piano keys, KEY is literally a key, CLOCK can have winding keys, BELL could be a stretch. COMPASS and TELESCOPE don't relate to keys. Go confident with 4.
→ Guesses: KEY, PIANO, CLOCK, BELL

---

## CURRENT GAME

**Available Words (unrevealed):** {{AVAILABLE_WORDS}}

**Board Layout:**
{{BOARD}}

**Current Clue:** {{CLUE}} ({{NUMBER}})

**List your guesses, one word per line. Most confident first. Only choose from available words above.**

