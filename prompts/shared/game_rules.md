# Codenames - Core Game Rules

## Game Overview
This is Codenames, a strategic word association game where two teams (Red and Blue) compete to identify all their agents while avoiding bystanders and the assassin. Codenames is one of the games in the BASED (Benchmark for Association, Sorting, and Entity Deduction) evaluation framework. There is a grid of 25 words, 8 allocated to Red, 8 allocated to Blue, 1 additional to the team that goes first (9 total), 7 which are innocent bystanders (neutral), and 1 assassin (instant loss).

## Turn Structure
1. **Spymaster Phase**: Current team's Spymaster provides a clue and number (see "Clue Validation Rules")
2. **Referee Phase**: Referee validates the clue for rule compliance
3. **Operative Phase**: Current team's Operatives make guesses (See "Guess Outcomes")
4. **Resolution**: Process guesses and determine turn outcome
5. **Team Switch**: If turn ends, check Win Conditions and if false, switch to other team

## Guess Outcomes
- **Own Agent**: ✓ Correct! Continue guessing (up to limit)
- **Enemy Agent**: ✗ Help enemy team, end turn immediately  
- **Bystander**: ○ Waste a guess, end turn immediately
- **Assassin**: 💀 Instant game loss for guessing team

## Win Conditions
- **Victory**: Find all your Agents first
- **Defeat**: Guess the Assassin (instant loss)
- **Defeat**: Enemy team finds all their Agents first

## Roles & Responsibilities

### Spymaster
- Has complete knowledge about all words on the board (the key card)
- Provides clues to guide their Operatives
- Must follow strict clue rules (see "Clue Validation Rules")

### Operatives  
- Only see the grid of 25 words
- Receive clues from their Spymaster
- Make guesses based on the clue
- Can guess up to **NUMBER + 1** words per turn (plus-one rule), with limited exceptions (See "Rules for Experts")

### Referee
- Validates that Spymaster clues comply with "Clue Validation Rules"
- Can reject invalid clues

## Clue Validation Rules

### Valid Clues Must:
1. **ALWAYS be a single word, with some limited exceptions** (no sentences or explanations)
2. **NEVER be a word currently on the grid** (exact match forbidden)
3. **NEVER be a direct variant of a word on the grid** (plural, past tense, etc.)
4. **NEVER reference the number of letters** in target words
5. **NEVER reference positions** on the board (row, column, location)

### Single Word Exceptions
1. English has three ways to write compound words. "Greenhouse" is one word. "Pack rat" is two words. "Mother-in-law" is hyphenated. Technically, only "Greenhouse" is a one-word clue. Compound words are allowed when describing specific ideas.
2. Proper Names are ALWAYS valid clues if they follow the other rules. Proper Names, such as "George Washington" or "The Three Musketeers" are valid words, as is "New York" or "The Big Apple".
3. Acronyms and Abbreviations are allowed when they are commonly referred to as a single word. Examples include CIA, UK, PhD and technical words like SQL, Radar, or Sonar.

### Penalty for Invalid Clue
If a spymaster gives an invalid clue, the team's turn ends immediately. As an additional penalty, the Referee reveals one of the opposing team's agents before the next turn begins.

## Rules for Experts
- **Expert Clue: Zero**: You are allowed to use 0 as the number part of your clue. For example, "Feathers (0)" means "None of our agents relate to 'Feathers'". If 0 is the number, the usual limit on guesses does not apply. Operatives can guess as many words as they want. They still must guess at least one word.
- **Expert Clue: Unlimited**: Sometimes you may have multiple unguessed agents related to your clues from previous turns. If you want your team to guess more than one of them, you may say unlimited instead of a number. The disadvantage is that the operatives do not know how many words are related to the new clue. The advantage is that they may guess as many words as they want.
