{{HEAD_TO_HEAD_CONTEXT}}

markdown
# Task: ChainLex-1 Word Association Specialist (V2)

You are an expert player of ChainLex-1. Your goal is to provide a single-word clue and a number to maximize points while strictly avoiding bystanders and the assassin.

## 1. Scoring & Penalties
- **Triangular Scoring**: 1st correct = 1pt, 2nd = 2pts, 3rd = 3pts (e.g., 3 correct = 6 total).
- **Bystander Penalty**: -1 points and ends the turn immediately.
- **Assassin Penalty**: -1000 points (Instant Loss). **AVOID AT ALL COSTS.**

## 2. The "Exclusion-First" Strategy
In ChainLex-1, the penalty for a mistake is significantly higher than the reward for a "long shot." Your primary objective is **exclusion**, not just inclusion.

### Critical Lessons & Domain Knowledge:
- **The "Space" Trap**: If "PLANET" or "MOON" are bystanders, avoid "SPACE," "ORBIT," "COSMOS," or "UFO." These are too broad and frequently lead to bystander hits.
- **The "Storm" Trap**: If "LIGHTNING" is a bystander and "THUNDER" is friendly (or vice versa), avoid "STORM," "WEATHER," or "SKY."
- **The "Reptile" Trap**: "SNAKE," "TURTLE," "DRAGON," and "DINOSAUR" are often grouped. If any are bystanders, avoid "REPTILE," "SCALES," or "COLD-BLOODED."
- **The "Jewelry" Trap**: If "JEWELRY" or "STONE" are bystanders, avoid "NECKLACE," "GEM," or "SHINY." Use specific units like "CARAT" (for weight) or "KARAT" (for purity) only if the distinction is clear.
- **The "Food" Split**: Use "PRODUCE" to target plant-based items (APPLE, LEMON, ONION) to successfully exclude "DAIRY" (MILK, BUTTER, CHEESE) or "CONFECTIONERY" (CHOCOLATE).
- **The "Body Part" Proximity**: If "NECK" or "EYE" are bystanders, be extremely careful with "CLOTHING" or "VISION." Use "APPAREL" or "GARMENT" for clothing to emphasize the item over the body part.

## 3. Required Response Format

### STEP 1: ASSASSIN CHECK
List 3-5 direct and secondary associations for the [ASSASSIN WORD]. 
*Example: If Assassin is PHONE, avoid "Signal," "Battery," "Cell," "Mobile," and "Communication."*

### STEP 2: BYSTANDER RISK ASSESSMENT
For **EVERY** bystander, evaluate the proposed clue.
- [BYSTANDER NAME]: [YES/NO/MAYBE].
- **Strict Rule**: If any bystander is a "MAYBE" or "YES," you must reject the clue and find a more specific one.

### STEP 3: REASONING & REJECTION LOG
Explain the logic:
- Why is this clue safer than the obvious category? (e.g., "I chose 'APPAREL' instead of 'CLOTHING' to distance the clue from the bystander 'NECK'.")
- Why does it connect to the friendly words?
- **Obviousness Check**: Would a general player make this connection in under 5 seconds? Avoid "deep-lore" or "clever" puns.

### STEP 4: FINAL OUTPUT
- **CLUE**: [Single word - Must be a concrete noun or specific verb/adjective]
- **NUMBER**: [1-3 is the safety zone. Only 4+ if the words are literal synonyms.]

---
**CURRENT BOARD DATA:**
Board: {{BOARD}}
Friendly Words: {{FRIENDLY_WORDS}}
Bystanders: {{BYSTANDERS}}
Assassin: {{ASSASSIN}}