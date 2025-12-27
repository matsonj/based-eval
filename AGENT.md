# Agent Instructions for BASED Eval

## Quick Reference

```bash
# Install & setup
uv sync
export OPENROUTER_API_KEY="your-key"

# Run games
uv run based codenames run --red gpt4 --blue claude
uv run based chainlex run --model-away gpt-4o --model-home gemini-3-flash
uv run based connections run --model gemini-flash --puzzles 10

# Tournament evaluation
uv run based chainlex eval --all --dry-run          # Preview schedule
uv run based chainlex eval --all                    # Full round-robin (16 threads)
uv run based chainlex eval --add-model new-model    # Add model to existing results

# DSPy optimization
uv run based chainlex optimize --model gemini-3-flash --num-train 50
uv run based chainlex deploy-prompts

# Development
uv run pytest
uv run black . && uv run isort .
```

## Architecture

### Codenames Game Flow
1. **Board Setup**: 25 words (9 red, 8 blue, 7 bystanders, 1 assassin)
2. **Turn Loop**: Teams alternate Spymaster → Operative phases
3. **Win**: Find all agents OR opponent hits assassin

### ChainLex-1 Game Flow
1. **Board Setup**: 16 words (8 friendly, 7 bystanders, 1 assassin)
2. **Away Turn**: First player gives clue + guesses (blind to opponent)
3. **Home Turn**: Second player knows opponent's score, adapts strategy
4. **Scoring**: Triangular (1+2+3+...), bystander=-5, assassin=-28
5. **Win**: Higher score wins; both hit assassin = tie

### Key Design Principles
- **Stateless AI Calls**: Each OpenRouter request independent
- **External Prompts**: All prompts in Markdown files (`prompts/`)
- **Home/Away Mechanic**: Second player advantage amplifies model intelligence differences
- **Append Mode**: `--add-model` appends to existing results.csv

## Code Conventions
- Type hints throughout
- `rich` for console output
- `typer` for CLI
- JSONL for structured logs

## Common Tasks

### Testing Prompts
```bash
uv run based codenames prompt spymaster --seed 42 --team red
uv run based chainlex prompt clue_giver --seed 42
```

### Adding Models
1. Edit `shared/inputs/model_mappings.yml`
2. Test: `uv run based codenames run --red NEW_MODEL --blue claude`

### Modifying Prompts
- Edit `prompts/*.md` files
- Template vars: `{{BOARD}}`, `{{CLUE}}`, `{{HEAD_TO_HEAD_CONTEXT}}`
- Test with `--verbose` flag

### Debugging
1. `--verbose` for full AI exchanges
2. Check `logs/` directory
3. Use `--seed` for reproducibility

## Terminology

| Term | Codenames | ChainLex-1 |
|------|-----------|------------|
| Clue giver | Spymaster | Clue Giver |
| Guesser | Operative | Guesser |
| Target word | Agent | Friendly Word |
| Neutral | Bystander (-0) | Bystander (-5) |
| Instant loss | Assassin | Assassin (-28) |
| Advantage | First turn | Home (2nd, knows score) |

## File Structure

```
codenames/          # Codenames game
chainlex/           # ChainLex-1 game
├── optimization/   # DSPy prompt optimization
├── prompts/        # Role prompts (clue_giver.md, guesser.md)
└── optimized_prompts/  # DSPy output
connections/        # Connections game
shared/             # Infrastructure (controllog, adapters, utils)
```

## Environment Variables
- `OPENROUTER_API_KEY`: Required for AI models
- `MOTHERDUCK_DB`: Optional, for analytics upload
