{{HEAD_TO_HEAD_CONTEXT}}

markdown
# Task: ChainLex-1 Word Association (Expert Mode - Version 4.0)

You are playing ChainLex-1, a high-stakes word association game. Your goal is to identify "Friendly" words on a board that match a given Clue and Number. You must prioritize accuracy over quantity, as a single mistake negates all progress for the turn.

## Scoring & Risks
- **Correct Guess**: Points increase exponentially (1st=1pt, 2nd=3pt, 3rd=6pt, etc.).
- **Bystander (Neutral)**: -1 points and IMMEDIATELY ends your turn.
- **Assassin**: -1000 points and IMMEDIATELY lose.
- **The "Wipeout" Rule**: One wrong guess (Bystander or Assassin) negates all previous gains in that turn. **Accuracy is 10x more important than quantity.**

## Core Strategy: The "Strict Literalism" Filter
1. **Container vs. Content**: Prioritize the **container** or **venue** (WAREHOUSE, GARAGE, PIANO) over the **content** or **output** (CAR, GOODS, SONG).
2. **Formal over Colloquial**: Stick to formal, legal, or technical definitions. Avoid slang or metaphorical "common phrases" (e.g., for "CLAUSE," use CONTRACT, not DEAL).
3. **Antonym Pairs**: Direct antonyms (e.g., LIFE and DEATH for "VITAL") are often valid Tier 1 links.
4. **The "Bystander" Trap**: 
   - **Generic Categories**: Words like GAME, MUSIC, or BATTLE are often traps if a more specific word (BOXING, OPERA) exists.
   - **Anatomical/Temporal terms**: BONE, MUSCLE, ANCIENT, and CENTURY are high-probability traps.
   - **Location vs. Event**: Do not pick an event (REVOLUTION) if the clue refers to a location (SQUARE).
   - **The "Thematic" Trap**: Avoid words that share a "vibe" but aren't technically defined by the clue (e.g., for "CHESS," KING is Tier 1, but KINGDOM is a trap).

## Domain-Specific Knowledge & Lessons Learned
- **STRUCTURE**: **FORTRESS** and **BRIDGE** are Tier 1. **CASTLE** is a known Bystander trap for this clue.
- **CHESS**: **KING**, **QUEEN**, and **KNIGHT** are Tier 1. **CASTLE** is acceptable as a 4th link (referring to the move/rook), but **CHAMPION** and **KINGDOM** are traps.
- **MORTAL**: **LIFE**, **DEATH**, and **HEART** are Tier 1. **DEFEAT** is a known Bystander trap.
- **CONFECTION**: **CAKE** and **SUGAR** are Tier 1.
- **HARDWARE**: **MOUSE** and **CIRCUIT** are Tier 1. **WIRE** is a high-risk Bystander.
- **SANCTUARY**: **TEMPLE** is Tier 1. **PALACE** and **CASTLE** are traps.
- **SQUARE**: **DANCE** and **BOXING** (The Squared Circle) are correct. **BATTLE** is a bystander.
- **VALKYRIE**: **VIKING** and **OPERA** (Wagner) are correct.
- **VITAL**: **LIFE**, **HEART**, and **DEATH** are correct.
- **STORAGE**: **WAREHOUSE** and **GARAGE** only.
- **CLAUSE**: **CONTRACT** only.
- **RECITAL**: **PIANO** only.
- **POLES**: **EARTH** and **MAGNET** only.
- **ORBIT**: **SATELLITE** and **MOON** only.
- **PULSE**: **HEART** and **LIFE** only.
- **LINK**: **CHAIN** only.
- **FACILITY**: **WAREHOUSE** and **GARAGE** are Tier 1. **FACTORY** is a known Bystander trap.
- **MALLET**: **HAMMER** and **PIANO** (internal mechanism) are Tier 1.
- **FIELD**: **MAGNET** and **EARTH** (Geomagnetic) are Tier 1. **PLANET** and **STAR** are high-risk Bystander traps.

## Decision Protocol
1. **Identify Candidates**: List all words related to the clue.
2. **The "One-Word" Threshold**: If the clue number is 2 or 3, but only ONE word is a "Tier 1" (unambiguous, literal) match, **you must stop and only guess that one word.** 
3. **The "Bystander" Test**: Ask: "Is this word a generic component (like WIRE) or a different category (like CASTLE for Sanctuary)?" If yes, exclude it.
4. **The "Specific vs. General" Rule**: If a clue could apply to multiple words but one is a specific technical definition (e.g., PIANO for MALLET) and others are general categories (e.g., MUSIC), choose the specific one.
5. **Rank by Confidence**: Order guesses from "Definitional" to "Thematic."

## Input Format
- **board**: A list of words currently available.
- **clue**: The one-word hint provided.
- **number**: The intended number of matches.

## Output Format
1. **Reasoning**: 
   - Explain why the chosen words are "Tier 1" literal matches.
   - Explicitly list rejected words and label them as "Bystander Traps" (e.g., "Generic Category," "Metaphorical Trap," or "Component Trap").
   - Justify why you are guessing the full "Number" or why you are stopping early (e.g., "Stopping at 2/3 to avoid high-risk Bystander traps like CASTLE or DEFEAT").
2. **Guesses**: List your guesses, one per line, most confident first.

---

## CURRENT GAME

**Available Words (unrevealed):** {{AVAILABLE_WORDS}}

**Board Layout:**
{{BOARD}}

**Current Clue:** {{CLUE}} ({{NUMBER}})

**List your guesses, one word per line. Most confident first. Only choose from available words above.**