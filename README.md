# BASED Eval

**Benchmark for Association, Sorting, and Entity Deduction**

A multi-game AI evaluation framework for testing language model capabilities through strategic word and deduction games.

## Games

### 🎯 Codenames

Strategic word association game where AI Spymasters give one-word clues to help AI Operatives identify their team's agents on a 5×5 grid.

```bash
uv run based codenames run --red gemini-flash --blue gemini-flash
```

- **25 words** in a 5×5 grid: 8-9 red agents, 8-9 blue agents, 7 bystanders, 1 assassin
- **Roles**: Spymaster (gives clues), Operative (guesses), Referee (validates clues)

### ⛓️ ChainLex-1

Fast head-to-head word association game designed for efficient AI evaluation. Two models compete on the same 16-word board with home/away advantage.

```bash
uv run based chainlex run --model-away gpt-4o --model-home gemini-3-flash
```

- **16 words**: 8 friendly, 7 bystanders, 1 assassin
- **Single round**: One clue + guesses per player
- **Home advantage**: Second player (home) knows opponent's score
- **Scoring**: Triangular (1+2+3+... max 36), bystander=-1, assassin=instant loss
- **DSPy optimization**: GEPA-based prompt evolution with puzzle pools
- **Puzzle pools**: Separate training (50) and eval (50) puzzles with semantic clustering

### 🔗 Connections

NYT-style puzzle where AI models identify four groups of four related words from a 16-word grid.

```bash
uv run based connections run --model gemini-flash --puzzles 10
```

- **16 words** → 4 groups of 4 with hidden categories
- **4 mistakes** allowed before game over
- **Qualification gate**: ≥50% win rate required for Codenames tournament

## Installation

```bash
git clone https://github.com/matsonj/based-eval
cd based-eval
uv sync
export OPENROUTER_API_KEY="your-key-here"
```

## Quick Start

```bash
# Codenames
uv run based codenames run --red gpt4 --blue claude
uv run based codenames run --red gpt4 --blue claude --interactive red-spymaster  # Human play

# ChainLex-1
uv run based chainlex run --model-away gpt-4o --model-home gemini-3-flash

# Connections
uv run based connections run --model gemini-flash --puzzles 5

# List models
uv run based codenames list-models
```

## Tournament Evaluation

### ChainLex-1 Tournament

```bash
# Preview matchup schedule and cost estimate
uv run based chainlex eval --all --dry-run

# Run full round-robin (16 threads, 4 games per matchup)
uv run based chainlex eval --all

# Add a new model to existing results (appends to results.csv)
uv run based chainlex eval --add-model new-model-name

# Cost estimation (runs each model once vs gemini-3-flash)
uv run based chainlex cost-estimate
```

**Schedule**: Each pair plays 4 games for balanced evaluation:
- 1 hard puzzle × 2 (home/away swap)
- 1 easy puzzle × 2 (home/away swap)

Output is Bradley-Terry compatible.

> **Note**: 4 games per matchup yields wide confidence intervals (~±75 rating points). For tighter CIs, increase games per pair (20+ recommended).

### Codenames Tournament

```bash
# List qualified models
uv run based codenames list-canonical

# Estimate cost
uv run based codenames cost-estimate --seed 42

# Generate schedule and run
uv run based codenames schedule --seed 42 --output logs/eval
uv run based codenames eval --schedule logs/eval/schedule.yml --threads 16

# Retry failed games
uv run based codenames retry --schedule logs/eval/schedule.yml
```

### Bradley-Terry Ranking

```bash
# Generate leaderboard from results (basic format)
uv run based analytics leaderboard -r logs/chainlex/eval/results.csv

# Use detailed_results.csv for home/away splits analysis
uv run based analytics leaderboard -r logs/chainlex/eval/detailed_results.csv
```

**Output**:
- `leaderboard.csv` - Full rankings with ratings and confidence intervals
- `leaderboard.png` - Forest plot visualization

**Features**:
- **Tie handling**: Ties map to 0.5 wins (per [arena-rank](https://github.com/lmarena/arena-rank) standard)
- **Home/Away splits**: Auto-detected when using `detailed_results.csv` (shows per-model home vs away performance)
- **Confidence intervals**: 95% CI via sandwich estimator
- **Baseline rating**: 1600 (matches standard ELO conventions)

## DSPy Prompt Optimization

ChainLex-1 includes GEPA (Genetic Evolution of Prompts Algorithm) for prompt optimization:

```bash
# Generate puzzle pools (50 training + 50 eval with semantic clustering)
uv run based chainlex generate-puzzles

# Optimize prompts (uses training pool only)
uv run based chainlex optimize --model gemini-3-flash --num-train 50

# Optimize with model blending (round-robin across multiple models)
uv run based chainlex optimize --model gemini-3-flash --blend

# Control optimization intensity
uv run based chainlex optimize --model gemini-3-flash --budget small  # light, medium, large, insane

# Deploy optimized prompts
uv run based chainlex deploy-prompts

# Rollback to originals
uv run based chainlex rollback-prompts

# List available puzzles
uv run based chainlex list-puzzles --pool training
```

**Puzzle Architecture**:
- Training pool (`chainlex/inputs/puzzles_training.yaml`): Used only by optimizer
- Eval pool (`chainlex/inputs/puzzles_eval.yaml`): Used by `eval` and `run` commands
- Semantic clustering ensures meaningful difficulty through word similarity metrics

## Analytics

```bash
uv run based analytics trial-balance   # Verify double-entry accounting
uv run based analytics cost-report     # Cost breakdown by model
uv run based analytics upload          # Upload logs to MotherDuck
```

## Available Models

80+ models via OpenRouter. Key families:

| Category | Models |
|----------|--------|
| **Thinking** | o3, gpt5, grok4, gemini-2.5, sonnet-4, opus-4, deepseek-r1 |
| **Standard** | gpt4o, claude, sonnet-3.5, grok3, llama-3.3, qwen3 |

## Project Structure

```
based-eval/
├── cli.py                      # Unified CLI entry point
├── codenames/                  # Codenames game
│   ├── cli_codenames.py        # CLI (run, schedule, eval, retry)
│   ├── game.py                 # Game logic
│   └── prompts/                # Role prompts
├── chainlex/                   # ChainLex-1 game
│   ├── cli_chainlex.py         # CLI (run, eval, optimize, cost-estimate)
│   ├── game.py                 # Game logic (home/away)
│   ├── game_engine.py          # Shared scoring/parsing (single source of truth)
│   ├── puzzle_generator.py     # Semantic clustering puzzle generation
│   ├── puzzle_loader.py        # Training/eval puzzle pool loader
│   ├── prompts/                # Role prompts
│   ├── inputs/                 # Puzzle pools and word data
│   │   ├── puzzles_training.yaml  # 50 training puzzles (optimizer only)
│   │   ├── puzzles_eval.yaml      # 50 eval puzzles (eval/run commands)
│   │   └── word_pool.yaml         # Curated words with semantic categories
│   └── optimization/           # DSPy optimization
├── connections/                # Connections game
│   └── src/connections_eval/   # Game logic and CLI
├── shared/                     # Infrastructure
│   ├── controllog/             # Double-entry logging SDK
│   ├── adapters/               # OpenRouter client
│   └── inputs/model_mappings.yml
└── logs/                       # Game logs
    ├── controllog/             # Unified analytics (all games)
    ├── chainlex/               # ChainLex game-specific logs
    └── eval/                   # Tournament results
```

## Shared Infrastructure

### controllog SDK

Double-entry accounting for structured event logging:

```python
from shared import controllog as cl

# IMPORTANT: Always use Path("logs") for unified analytics
cl.init(project_id="codenames", log_dir=Path("logs"))
cl.model_prompt(task_id=task_id, agent_id="spymaster", request_text=prompt, ...)
cl.model_completion(task_id=task_id, wall_ms=latency, cost_money=cost, response_text=response, ...)
cl.state_move(task_id=task_id, from_="WIP", to="DONE")
cl.game_complete(task_id=task_id, game_id=game_id, outcome="model_away", ...)  # Game summary
```

**Event Types**:
- `model_prompt` / `model_completion` - AI calls with tokens, cost, timing, and full request/response text
- `state_move` - State transitions (NEW→WIP→DONE)
- `game_complete` - Game-level summary for leaderboards and analytics (ChainLex)

The SDK writes to `logs/controllog/<YYYY-MM-DD>/events.jsonl` and `postings.jsonl`. All games must use `Path("logs")` (not game-specific paths) so `uv run based analytics upload` can find all data.

### MotherDuck Integration

```bash
export MOTHERDUCK_DB="md:based_eval"
uv run based analytics upload
uv run based analytics trial-balance
```

## Development

```bash
# Run all tests
uv run pytest tests/

# Run specific game tests
uv run pytest tests/test_chainlex_game.py tests/test_chainlex_player.py -v  # ChainLex
uv run pytest tests/test_game.py -v                                         # Codenames
uv run pytest tests/test_controllog.py -v                                   # Controllog SDK

# Format
uv run black . && uv run isort .
```

### Test Coverage
- **ChainLex**: 55 tests (game logic, player parsing, controllog events)
- **Codenames**: 7 tests (game logic)
- **Controllog**: 9 tests (event builders, text inclusion)
- **Metadata/Adapter**: 6 tests (cost extraction, metadata storage)

## License

MIT License

## Credits

- Codenames by Vlaada Chvátil. [Official Rules](https://czechgames.com/files/rules/codenames-rules-en.pdf)
- Connections puzzle by The New York Times
