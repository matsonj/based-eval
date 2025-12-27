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

# Analytics / Leaderboard
uv run based analytics leaderboard -r logs/chainlex/eval/detailed_results.csv

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
4. Query controllog events for full request/response text:
   ```sql
   SELECT payload_json.request_text, payload_json.response_text
   FROM controllog.events WHERE kind = 'model_prompt' OR kind = 'model_completion'
   ```

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
logs/
├── controllog/     # Unified analytics (ALL games write here)
│   └── YYYY-MM-DD/ # Date-partitioned events.jsonl + postings.jsonl
├── chainlex/       # ChainLex game-specific logs (box_scores, etc.)
├── codenames/      # Codenames game-specific logs
└── eval/           # Tournament results
```

## Controllog Convention

**IMPORTANT**: All games must initialize controllog with `Path("logs")` (the top-level logs directory):

```python
cl.init(project_id="chainlex", log_dir=Path("logs"))  # ✅ Correct
cl.init(project_id="chainlex", log_dir=log_dir)       # ❌ Wrong if log_dir is game-specific
```

The SDK automatically creates `logs/controllog/<date>/` subdirectories. This ensures `uv run based analytics upload` finds all game data without needing `--log-path` flags.

Game-specific files (box_scores, game_metadata, play_by_play) can still go to game-specific directories.

### Event Types

| Event | Purpose | Key Fields |
|-------|---------|------------|
| `model_prompt` | AI request | `request_text`, `prompt_tokens`, `model` |
| `model_completion` | AI response | `response_text`, `completion_tokens`, `wall_ms`, `cost_money` |
| `state_move` | State transition | `from_`, `to` (NEW→WIP→DONE) |
| `game_complete` | Game summary | `outcome`, `winner_model`, `scores`, `wall_ms` |

### game_complete Event (ChainLex)

Emitted at game end for leaderboards and analytics:

```python
cl.game_complete(
    task_id="game:abc123",
    game_id="abc123",
    model_away="claude-3",
    model_home="gpt-4",
    outcome="model_away",  # or "model_home" or "tie"
    winner_model="claude-3",
    score_away=15,
    score_home=10,
    margin=5,
    correct_guesses_away=4,
    correct_guesses_home=3,
    total_guesses=8,
    wall_ms=45000,
    cost_money=0.01,
)
```

This enables:
- **Leaderboards**: Query by outcome/winner_model
- **Head-to-head matrices**: Group by (model_away, model_home) pairs  
- **Efficiency metrics**: cost_money/correct_guesses, wall_ms/total_guesses

> **TODO**: Add game_complete events to Codenames (#4) and Connections (#5)

## Analytics Leaderboard

The `analytics leaderboard` command generates Bradley-Terry ratings using [arena-rank](https://github.com/lmarena/arena-rank).

### Data Formats

| File | Columns | Features |
|------|---------|----------|
| `results.csv` | `model_a, model_b, winner` | Basic BT ratings |
| `detailed_results.csv` | `model_home, model_away, winner, ...` | BT ratings + home/away splits |

### Tie Handling

Ties are handled per arena-rank defaults:
- `winner="tie"` → outcome 0.5 (half-win for each model)
- `winner="model_a"` → outcome 1.0 (model_a wins)
- `winner="model_b"` → outcome 0.0 (model_b wins)

Win% in tables uses `(wins + 0.5*ties) / total_games` to match BT model.

### Key Parameters

```python
# In shared/cli_analytics.py
BradleyTerry(n_competitors=len(dataset.competitors), init_rating=1600)
```

- `init_rating=1600`: Baseline rating (matches standard ELO)
- Variance will be high with few games per matchup (aim for 20+ per pair)

### Output Files

- `leaderboard.csv`: Full rankings with CI
- `leaderboard.png`: Forest plot (dot + CI range, color-coded by tier)

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test files
uv run pytest tests/test_chainlex_game.py -v      # ChainLex game logic
uv run pytest tests/test_chainlex_player.py -v    # ChainLex player/parsing
uv run pytest tests/test_controllog.py -v         # Controllog SDK
uv run pytest tests/test_game.py -v               # Codenames game logic
```

### Test Structure

| File | Tests | Coverage |
|------|-------|----------|
| `test_chainlex_game.py` | 24 | Board setup, scoring, clue validation, controllog |
| `test_chainlex_player.py` | 22 | Response parsing, metadata storage, board formatting |
| `test_controllog.py` | 9 | game_complete, model_prompt/completion text |
| `test_game.py` | 7 | Codenames game logic |
| `test_metadata_logging.py` | 6 | Adapter cost extraction |

## Environment Variables
- `OPENROUTER_API_KEY`: Required for AI models
- `MOTHERDUCK_DB`: Optional, for analytics upload
