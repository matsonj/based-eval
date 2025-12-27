# Agent Instructions for BASED Eval

## Overview
BASED (Benchmark for Association, Sorting, and Entity Deduction) is a multi-game AI evaluation framework:
- **Codenames**: Strategic word association—teams use AI spymasters/operatives to find agents while avoiding the assassin
- **Connections**: NYT-style puzzle—group 16 words into 4 categories of 4. Uses canonical 20-puzzle set (v3.0.0+)

## Commands

```bash
# Core
uv sync                                                    # Install dependencies
uv run pytest                                              # Run tests
uv run black . && uv run isort .                           # Format code
uv run mypy codenames/                                     # Type check

# Codenames
uv run based codenames run --red gpt4 --blue claude        # Run game
uv run based codenames run ... --interactive red-spymaster # Interactive mode
uv run based codenames prompt spymaster --seed 42          # Test prompts
uv run based codenames cost-estimate --output logs/eval    # Estimate costs
uv run based codenames schedule --output logs/eval         # Generate schedule
uv run based codenames eval --schedule logs/eval/schedule.yml  # Run tournament
uv run based codenames validate-models                     # Validate API access
uv run based codenames optimize --log-dir logs/            # DSPy prompt optimization
uv run based codenames export-prompts                      # Export optimized prompts

# Connections
uv run based connections run --model gpt4 --canonical      # Run 20 canonical puzzles
uv run based connections rank --threads 4                  # Rank puzzle difficulty
uv run based connections list-puzzles --difficulty         # List puzzles

# Analytics
uv run based analytics trial-balance
uv run based analytics leaderboard --results logs/eval/results.csv
```

## Project Structure
```
codenames/              # Game package
├── cli*.py             # CLI commands (main, codenames, connections, analytics)
├── game.py             # Game logic, board management
├── player.py           # AIPlayer/HumanPlayer classes
├── prompt_manager.py   # Markdown template loader
├── adapters/           # OpenRouter API client
├── optimization/       # DSPy prompt optimization pipeline
└── utils/logging.py

connections/            # Puzzle game
├── src/connections_eval/{cli,core}.py  # CLI and game engine
└── inputs/             # puzzles.yml, difficulty.yml, prompt_template.xml

shared/                 # Infrastructure
├── controllog/         # Double-entry accounting SDK
├── adapters/           # Shared API adapters
└── utils/              # retry, timing, tokens, motherduck

inputs/                 # names.yaml (word bank), model_mappings.yml
prompts/                # Markdown templates: {red,blue}_{spymaster,operative}.md, referee.md, shared/game_rules.md

logs/                   # Output
├── based_*.log         # Debug logs
├── play_by_play_*.log  # Game events
├── box_scores_*.jsonl  # Team performance
├── game_metadata_*.jsonl  # AI metrics (tokens, costs, latency) - also used as DSPy training data
└── referee_*.log       # Clue validation

optimized_prompts/      # DSPy optimization output
├── optimized_pipeline.json  # Trained DSPy pipeline with demos
└── optimization_results.json  # Baseline vs optimized scores
```

## Code Style
Type hints throughout · PEP 8 · **rich** for console · **typer** for CLI · **pydantic** for validation · JSONL for structured logs

## Architecture

### Codenames Flow
1. **Board**: 25 words → 9 red, 8 blue, 7 bystanders, 1 assassin
2. **Turns**: Alternate Spymaster (clue + number) → Operative (up to N+1 guesses)
3. **Win**: Find all agents OR opponent hits assassin

### Design Principles
- **Stateless AI**: Each OpenRouter request independent (security)
- **External Prompts**: Markdown files for easy tuning
- **Flexible Players**: Same interface for AI/Human
- **Comprehensive Logging**: Human-readable + JSONL
- **Quiet Mode**: Tournaments use `quiet=True` for summaries only

### Testing
Unit tests (game logic) · Integration tests (AI interactions) · Mock API calls · AI vs AI and Human vs AI modes

## Environment
`OPENROUTER_API_KEY` required. If unset error, run `source .env` first.

## Development Tasks

### Testing Prompts
```bash
uv run based codenames prompt spymaster --seed 42 --team red
uv run based codenames prompt operative --seed 42 --clue "TOOLS" --number 3
uv run based codenames prompt referee --seed 42 --clue "WEAPONS" --number 2
uv run based codenames prompt operative --clue "ANIMALS" --number 0  # or --number unlimited
```

### Adding Models
1. Update `shared/inputs/model_mappings.yml`
2. Test: `uv run based codenames run --red NEW_MODEL --blue claude`
3. Validate: `uv run based codenames validate-models`

### Canonical Models
Models in `canonical_models` list participate in tournaments. Requirements: ≥50% Connections win rate + pass API validation (no 404s)

### Modifying Rules
Edit `game.py`: board setup, `process_guess()` win/lose conditions, N+1 rule

### Customizing Prompts
Edit `prompts/*.md` · Variables: `{{BOARD}}`, `{{CLUE}}`, etc. · Test with `--verbose`

### Prompt Optimization (DSPy)
Prompts can be optimized using DSPy to improve game performance.

```bash
# Run optimization (uses game logs as training data)
uv run based codenames optimize --log-dir logs/ --model gemini-3-flash

# Export optimized pipeline to game prompts
uv run based codenames export-prompts

# Test with a game
uv run based codenames run --red gemini-3-flash --blue gemini-3-flash --num-games 1
```

**Architecture** (`codenames/optimization/`):
- `data_extractor.py` - Extracts training data from `game_metadata_*.jsonl`
- `metrics.py` - Scoring: +1 correct, -0.5 bystander, -1 enemy, -999 assassin
- `modules.py` - DSPy modules for Spymaster/Operative with `ChainOfThought`
- `pipeline.py` - Joint optimization pipeline with rule-based referee (-1 penalty for invalid clues)
- `referee.py` - Deterministic clue validation (no LLM calls during training)
- `export_prompts.py` - Converts `optimized_pipeline.json` → markdown prompts

**Key Insights**:
- **Joint optimization**: Optimize Spymaster + Operative together (not separately)—they're interdependent
- **Rule-based referee in training**: Use deterministic validation during optimization to avoid LLM costs
- **Metric design**: Heavily penalize assassin (-999) to train risk aversion

**Critical Prompt Variables** (must be populated from `board_state`):
| Variable | Purpose |
|----------|---------|
| `{{CLUE_HISTORY}}` | Previous clues—**prevents repetition** |
| `{{REVEALED}}` | Already-revealed words—**prevents invalid references** |
| `{{BLUE_REMAINING}}` / `{{RED_REMAINING}}` | Remaining agent counts |
| `{{BOARD}}` | Unrevealed words only |

⚠️ **Common Bug**: If models repeat the same clue every turn, check that `player.py` passes actual `board_state.get("clue_history")` instead of a static placeholder string.

### Debugging
`--verbose` for AI exchanges · Check `logs/` · `--seed` for reproducibility

## Terminology
| Term | Description |
|------|-------------|
| Spymaster | Gives clues to guide operatives |
| Operative | Makes guesses based on clues |
| Agent | Team member to find |
| Bystander | Neutral word |
| Assassin | Instant loss if guessed |

## Cost Tracking

### True Cost = cost + upstream_cost
```python
total_cost = result.get("cost", 0) + result.get("upstream_cost", 0)
```
Some models (o-series, gpt5.x) report most cost in `upstream_cost`:
- **o3**: cost $0.007, upstream $0.24 → **$0.24**
- **gpt5.2-pro**: cost $0.01, upstream $2.13 → **$2.14**

### Thinking Model Tokens
Internal reasoning can use massive tokens:
- kimi-k2-thinking: avg 11K, max 23K
- gemini-3-pro: avg 8K, max 18K
- deepseek-r1: avg 6K, max 9K

**Warning**: Low `max_completion_tokens` (e.g., 8000) can exhaust tokens on thinking → empty output (`finish_reason: length`)

## Tournament Evaluation

### Bradley-Terry
- **2 games per matchup** (1 home, 1 away) sufficient for estimates
- ~40+ total games per model for stable rankings
- Full round-robin ensures comparability
- With 24 models: 276 matchups × 2 = **552 games**, 46 games/model

### Commands
- **Cost estimate**: Plays each canonical model vs gemini-3-flash, outputs `cost_estimate_results.csv`
- **Leaderboard**: Computes Bradley-Terry ratings, outputs `leaderboard.csv` + `leaderboard.png`
- **Validate**: Tests API access for all canonical models, reports errors

## Model Configuration

### Referee
Two-stage clue validation: `gemini-3-flash` (initial) → `gpt5.2` (if flagged invalid). Different models avoid agreement bias.

### API Retries
3 attempts with exponential backoff. Don't increase—more rarely helps.

### Model Types (in `model_mappings.yml`)
**Thinking** (o3, o4-mini, gpt5.x, gemini-2.5-pro, deepseek-r1): No temperature, 600s+ timeout, may have `reasoning` field

**Non-thinking** (gpt4o, claude-3.5-sonnet, llama-3.3): Standard temperature/max_tokens/timeout

## Connections Evaluation

### Puzzle Distribution (Fixed v3.0.0)
**Problem**: Time-based seeding gave different models different puzzles, invalidating comparisons.
**Solution**: `canonical: true` field + `--canonical` flag → 20 hardest puzzles, identical across models.

### Difficulty Ranking
`rank` command: 10 runs/puzzle on gemini-3-flash → score = `(win_rate * 100) - token_penalty` → outputs `puzzle_difficulty.yml`

**Tiers**: Hard (<50%), Medium (50-80%), Easy (≥80%)

### Canonical Puzzles (v3.0.0)
```
Hard (0%):      830, 813, 841, 820, 831
Low (10-30%):   842, 351, 835, 814, 246, 832, 822, 475
Medium (50-70%): 817, 840, 828, 825, 821, 837, 834, 304
```

### Parallel Execution
Default 8 threads · `--threads 1` for debugging · Interactive always sequential

### Versions
- **v3.0.0**: Puzzle ranking, canonical sets, parallel execution
- **v2.0.2**: Fixed reasoning field extraction

## TODOs
- [x] Model listing, config file support, expert clues, logging, tournaments, puzzle distribution fix, difficulty ranking
- [x] DSPy prompt optimization pipeline with joint Spymaster/Operative training
- [ ] Add more games · Improve AI response parsing · Game replay from JSONL

## Dependencies
typer (CLI) · rich (terminal) · openai (API) · pyyaml (config) · tenacity (retry) · pydantic (validation) · arena-rank (Bradley-Terry) · matplotlib (charts) · dspy (prompt optimization) · nltk (text processing)
