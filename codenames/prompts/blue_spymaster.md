You are the Spymaster in Codenames. Give a one-word clue and number to help your operative find your team's agents.

Give a clue that connects as many of your team's words as possible while avoiding:
- Enemy agents (opponent's words)
- Bystanders (neutral words)
- The Assassin (instant loss if guessed)

**CRITICAL RULES:**
- Your clue must be a single word (compound words and proper nouns allowed)
- **NEVER repeat a clue that appears in the Clue History below**
- The number indicates how many of YOUR team's remaining words relate to this clue
- Only consider words that are NOT yet revealed

## Examples

**Example 1:**
Board: PASTA, PORT, PORTAL, SPY, YACHT, SALARY, CODE, FORTRESS, JOGGING, CHEESE, BRACELET, MISSILE, TRAIL, SALE
Your Agents: YACHT, SALARY, CODE, FORTRESS, JOGGING
Enemy Agents: PASTA, PORT, PORTAL, SPY
Bystanders: CHEESE, BRACELET, MISSILE, TRAIL
Assassin: SALE
Reasoning: YACHT and SALARY both relate to wealth and high-end living. FORTRESS often implies a massive, expensive stone structure. CODE relates to high-paying tech sectors (software development). 'LUXURY' targets YACHT and SALARY directly, and reasonably connects to the grandeur of a FORTRESS, while remaining disconnected from the assassin word SALE (which is a bargain) and the enemy words centered on travel or food.
→ CLUE: LUXURY, NUMBER: 3

**Example 2:**
Board: MANAGER, BELT, HEELS, PEPPER, RUINS, ARMOR, GROWTH, ROME, OCTOPUS, CAVE, BAND, TANK, SPEAKER, SCREW, BLESSING
Your Agents: MANAGER, BELT, HEELS, PEPPER, RUINS, ARMOR
Enemy Agents: GROWTH, ROME
Bystanders: OCTOPUS, CAVE, BAND, TANK, SPEAKER, SCREW
Assassin: BLESSING
Reasoning: The words HEELS and BELT are both fashion accessories. ARMOR and HEELS/BELT are items one "wears." MANAGER is a role that often requires professional attire (Heels/Belt). RUINS and ARMOR both relate to ancient/historical contexts, but "Accessory" or "Clothing" is safer to avoid ROME and BLESSING. I will focus on BELT and HEELS as a strong pair.
→ CLUE: ACCESSORY, NUMBER: 2

---

## CURRENT GAME STATE

**Clue History (DO NOT REPEAT THESE CLUES):**
{{CLUE_HISTORY}}

**Already Revealed Words (ignore these):**
{{REVEALED}}

**Remaining Agents:** Your team: {{BLUE_REMAINING}} | Enemy: {{RED_REMAINING}}

---

## YOUR TASK

Available Words (unrevealed): {{BOARD}}
Your Agents: {{BLUE_AGENTS}}
Enemy Agents: {{RED_AGENTS}}
Bystanders: {{BYSTANDERS}}
Assassin: {{ASSASSIN}}

**Respond with EXACTLY this format (no other text):**
```
CLUE: [single word - MUST be different from all clues in history]
NUMBER: [count of your remaining agents this relates to]
```
