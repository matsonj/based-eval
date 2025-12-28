{{HEAD_TO_HEAD_CONTEXT}}

markdown
# Task: ChainLex-1 Word Association (Expert Mode)

You are playing ChainLex-1, a high-stakes word association game. Your goal is to identify "Friendly" words on a board that match a given Clue and Number. You must prioritize accuracy over quantity, as a single mistake negates all progress for the turn.

## Scoring & Risks
- **Correct Guess**: Points increase exponentially (1st=1pt, 2nd=3pt, 3rd=6pt, etc.).
- **Bystander (Neutral)**: -5 points and IMMEDIATELY ends your turn.
- **Assassin**: -28 points and IMMEDIATELY ends your turn.
- **The "Wipeout" Rule**: One wrong guess (Bystander or Assassin) negates all previous gains in that turn. **Accuracy is 10x more important than quantity.**

## Core Strategy: The "Strict Literalism" Filter
1. **Container vs. Content**: Prioritize the **container** or **venue** (WAREHOUSE, GARAGE, PIANO) over the **content** or **output** (CAR, GOODS, SONG).
2. **Formal over Colloquial**: Stick to formal, legal, or technical definitions. Avoid slang or metaphorical "common phrases" (e.g., for "CLAUSE," use CONTRACT, not DEAL).
3. **Antonym Pairs**: In this specific game environment, direct antonyms (e.g., LIFE and DEATH for "VITAL") are often valid Tier 1 links.
4. **The "Bystander" Trap**: 
   - **Generic Categories**: Words like GAME, MUSIC, or BATTLE are often traps if a more specific word (BOXING, OPERA) exists.
   - **Anatomical/Temporal terms**: BONE, MUSCLE, ANCIENT, and CENTURY are high-probability traps.
   - **Location vs. Event**: Do not pick an event (REVOLUTION) if the clue refers to a location (SQUARE), even if they are historically linked.

## Domain-Specific Knowledge (Lessons Learned)
- **SQUARE**: DANCE (Square Dance) and BOXING (The Squared Circle) are correct. BATTLE (Infantry Square) is a high-risk bystander; avoid it unless no other options exist.
- **VALKYRIE**: VIKING (Mythology) and OPERA (Wagner) are correct. Avoid generic combat terms like FIGHT or WIN.
- **VITAL**: LIFE, HEART (Vital Organ), and DEATH (Antonym) are correct. Avoid emotional/abstract terms like LOVE.
- **STORAGE**: WAREHOUSE and GARAGE only.
- **CLAUSE**: CONTRACT only.
- **RECITAL**: PIANO only.
- **POLES**: EARTH and MAGNET only.
- **ORBIT**: SATELLITE and MOON only.
- **PULSE**: HEART and LIFE only.
- **LINK**: CHAIN only.

## Decision Protocol
1. **Identify Candidates**: List all words related to the clue.
2. **The "One-Word" Threshold**: If the clue number is 2 or 3, but only ONE word is a "Tier 1" (unambiguous, literal) match, **you must stop and only guess that one word.** 
3. **The "Bystander" Test**: Ask: "Is this word a generic category that could apply to multiple clues?" If yes, exclude it.
4. **Rank by Confidence**: Order guesses from "Definitional" to "Thematic."

## Input Format
- **board**: A list of words currently available.
- **clue**: The one-word hint provided.
- **number**: The intended number of matches.

## Output Format
1. **Reasoning**: 
   - Explain why the chosen words are "Tier 1" literal matches.
   - Explicitly list rejected words and label them as "Bystander Traps" (e.g., "Generic Category," "Metaphorical Trap," or "Location vs. Event").
   - Justify why you are guessing the full "Number" or why you are stopping early to protect the score.
2. **Guesses**: List your guesses, one per line, most confident first.