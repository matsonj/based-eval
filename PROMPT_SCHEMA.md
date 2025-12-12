# Prompt Template Schema

This document defines the expected template variables for each prompt template to ensure proper hydration and prevent runtime errors.

## Template Variable Conventions

- All template variables use `{{VARIABLE_NAME}}` format
- Variable names are UPPERCASE with underscores
- The PromptManager automatically converts lowercase context keys to uppercase placeholders
- Example: context key `"clue"` becomes placeholder `{{CLUE}}`

## Spymaster Templates

### red_spymaster.md & blue_spymaster.md
**Required Variables:**
- `{{BOARD}}` - Game board as 5x5 grid (list of 25 words)
- `{{REVEALED}}` - Comma-separated list of revealed word names
- `{{TEAM}}` - Team name ("red" or "blue")
- `{{RED_REMAINING}}` - Number of red agents remaining
- `{{BLUE_REMAINING}}` - Number of blue agents remaining
- `{{RED_AGENTS}}` - Comma-separated list of red team agents
- `{{BLUE_AGENTS}}` - Comma-separated list of blue team agents
- `{{BYSTANDERS}}` - Comma-separated list of innocent bystanders
- `{{ASSASSIN}}` - The assassin word
- `{{CLUE_HISTORY}}` - Game history of previous clues and outcomes

**Context Keys (from code):**
```python
{
    "board": board_state["board"],
    "revealed": ", ".join(revealed_words),
    "team": self.current_team,
    "red_remaining": red_remaining,
    "blue_remaining": blue_remaining,
    "red_agents": ", ".join(red_agents),
    "blue_agents": ", ".join(blue_agents),
    "bystanders": ", ".join(bystanders),
    "assassin": ", ".join(assassin),
    "clue_history": clue_history
}
```

## Operative Templates

### red_operative.md & blue_operative.md
**Required Variables:**
- `{{BOARD}}` - Game board as 5x5 grid
- `{{AVAILABLE_WORDS}}` - Comma-separated list of unrevealed words
- `{{CLUE}}` - Current clue word from spymaster
- `{{NUMBER}}` - Number of words related to clue (int or "unlimited")
- `{{TEAM}}` - Team name ("red" or "blue")
- `{{CLUE_HISTORY}}` - Game history of previous clues and outcomes

**Context Keys (from code):**
```python
{
    "board": board_state["board"],
    "available_words": ", ".join(available_words),
    "clue": clue,
    "number": number,
    "team": team,
    "clue_history": clue_history
}
```

## Referee Template

### referee.md
**Required Variables:**
- `{{CLUE}}` - Proposed clue word to validate
- `{{NUMBER}}` - Proposed number (int or "unlimited")
- `{{TEAM}}` - Team making the clue ("red" or "blue")
- `{{BOARD}}` - Current board state as comma-separated list
- `{{TEAM_AGENTS}}` - Comma-separated list of current team's agents

**Context Keys (from code):**
```python
{
    "clue": clue,
    "number": parsed_number,
    "team": team,
    "board": ", ".join(board_state["board"]),
    "team_agents": ", ".join(team_agents)
}
```

## Shared Includes

### shared/game_rules.md
This file is included in other templates via `{{include:shared/game_rules.md}}` and contains no template variables itself.

## Validation

The PromptManager now includes strict validation that:
1. Identifies all template variables in loaded templates using regex `\{\{([A-Z_]+)\}\}`
2. Checks that all template variables have corresponding context values
3. Raises `PromptHydrationError` if any variables are missing
4. Causes the program to fail fast rather than continue with malformed prompts

## Adding New Templates

When adding new prompt templates:
1. Use consistent variable naming following the conventions above
2. Document required variables in this schema
3. Test with validation enabled to ensure all variables are properly hydrated
4. Use consistent Codenames terminology (spymaster, operative, clue, guess, agent, bystander, assassin)
