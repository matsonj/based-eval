You are playing ChainLex-1, a word association game. Give a one-word clue and number to maximize your score.

## Scoring
- Each correct guess earns points: 1st=1pt, 2nd=2pt, 3rd=3pt, etc. (triangular scoring)
- Maximum possible: 36 points (all 8 friendly words: 1+2+3+4+5+6+7+8)
- Bystander: -5 points and ENDS your turn
- Assassin: -999 points and ENDS your turn

## Your Goal
Give a clue that connects as many friendly words as possible while AVOIDING:
- Bystanders (neutral words) - penalty and turn ends
- The Assassin (instant catastrophic loss)

**Risk vs Reward**: 
- Safe 3-word clue = potential 6 points (1+2+3)
- Aggressive 6-word clue = potential 21 points (1+2+3+4+5+6) but higher risk

## Examples

**Example 1:**
Board: PASTA, PORT, SPY, YACHT, SALARY, CODE, FORTRESS, CHEESE, BRACELET, TRAIL, LION, WOLF, EAGLE, SHARK, TIGER, BEAR
Friendly: LION, WOLF, EAGLE, SHARK, TIGER, BEAR, YACHT, CODE
Bystanders: PASTA, PORT, FORTRESS, CHEESE, BRACELET, TRAIL, SALARY
Assassin: SPY
Reasoning: LION, WOLF, EAGLE, SHARK, TIGER, BEAR are all animals. This connects 6 friendly words safely, avoiding the assassin SPY and the bystanders. YACHT and CODE don't fit this theme but could be targeted with a second clue concept. "PREDATOR" connects 6 words for a potential 21 points.
→ CLUE: PREDATOR, NUMBER: 6

**Example 2:**
Board: PIANO, MIRROR, CROWN, DIAMOND, CLOCK, BELL, BOOK, TELESCOPE, COMPASS, ANCHOR, KEY, LAMP, COFFEE, HONEY, SALT, APPLE
Friendly: PIANO, MIRROR, CROWN, DIAMOND, CLOCK, BELL, KEY, LAMP
Bystanders: BOOK, TELESCOPE, COMPASS, ANCHOR, COFFEE, HONEY, SALT
Assassin: APPLE
Reasoning: PIANO, CLOCK, BELL, and KEY all have "keys" - piano keys, clock keys, bell keys (to wind them), and literally KEY. CROWN and DIAMOND are both associated with royalty/jewelry. A safe play is "KEYS" for 4 words (10 points potential). Avoid APPLE at all costs.
→ CLUE: KEYS, NUMBER: 4

---

## CURRENT GAME

**Board (16 words):** {{BOARD}}

**Your Friendly Words ({{NUM_FRIENDLY}}):** {{FRIENDLY_WORDS}}
**Bystanders:** {{BYSTANDERS}}
**Assassin:** {{ASSASSIN}}

**Respond with EXACTLY this format (no other text):**
```
CLUE: [single word]
NUMBER: [how many friendly words this clue connects to]
```

