# BASED Eval

**Benchmark for Association, Sorting, and Entity Deduction**

A multi-game AI evaluation framework for testing language model capabilities through strategic word and deduction games.

## Overview

BASED Eval tests AI models on:
- **Association**: Finding semantic connections between concepts
- **Sorting**: Categorizing and grouping related items
- **Entity Deduction**: Reasoning about hidden information and making inferences

## Games

### 🎯 Codenames

A strategic word association game where AI Spymasters give one-word clues to help AI Operatives identify their team's agents on a 5×5 grid.

```bash
uv run based codenames run --red gemini-flash --blue gemini-flash
```

**Game Setup:**
- 25 words in a 5×5 grid
- Red Team: 8-9 Agents | Blue Team: 8-9 Agents
- 7 Innocent Bystanders | 1 Assassin (instant loss)

**Roles:**
- **Spymaster**: Gives one-word clues with a number
- **Operative**: Guesses words based on clues
- **Referee**: Validates clue legality (always `gemini-3-flash`)

### 🔗 Connections

Based on the New York Times Connections puzzle. AI models must identify four groups of four related words from a 16-word grid.

```bash
uv run based connections run --model gemini-flash --puzzles 10
```

**Game Setup:**
- 16 words to sort into 4 groups of 4
- Each group has a hidden category (e.g., "Types of keys", "Words before 'dog'")
- 4 mistakes allowed before game over

**Qualification Gate:** Models must achieve ≥50% win rate on Connections to qualify for Codenames tournament evaluation.

## Features

- **Multi-Game Framework**: Unified infrastructure for multiple evaluation games
- **AI vs AI**: Pit different models against each other
- **Human vs AI**: Interactive mode for human players
- **200+ Models**: OpenRouter integration for broad model coverage
- **Thinking Model Support**: Automatic detection and configuration for reasoning models (o3, gpt-5, grok-4, etc.)
- **Structured Logging**: controllog SDK with double-entry accounting
- **MotherDuck Integration**: Upload results for analytics
- **Cost Tracking**: Token usage and API cost tracking per call
- **Tournament Evaluation**: Round-robin scheduling with parallelism and Bradley-Terry ranking output
- **Reproducibility**: Seed-based deterministic board generation
- **Canonical Models**: Qualification gate via Connections eval for tournament participation

## Installation

Requires Python ≥3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/matsonj/based-eval
cd based-eval
uv sync
```

## Quick Start

### Set Up API Key
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### Run Codenames (AI vs AI)
```bash
uv run based codenames run --red gpt4 --blue claude
```

### Run Codenames (Interactive)
```bash
uv run based codenames run --red gpt4 --blue claude --interactive red-spymaster
```

### Run Connections
```bash
uv run based connections run --model gemini-flash --puzzles 5
```

### List Available Models
```bash
uv run based codenames list-models
uv run based connections list-models
```

### Analytics
```bash
uv run based analytics trial-balance
uv run based analytics cost-report
uv run based analytics leaderboard
```

## Tournament Evaluation

BASED Eval includes a full tournament system for evaluating models head-to-head with Bradley-Terry ranking.

### Qualification

Models must achieve ≥50% win rate on Connections eval to qualify for tournament participation. Qualified models are flagged as `canonical_models` in `shared/inputs/model_mappings.yml`.

```bash
# List canonical models for tournament
uv run based codenames list-canonical
```

### Cost Estimation

Before running a full tournament, estimate costs by running each model once against `gemini-3-flash`:

```bash
# Run cost estimation (1 game per canonical model vs gemini-3-flash)
uv run based codenames cost-estimate --seed 42 --output logs/eval --threads 8
```

This command:
1. Runs each canonical model once against `gemini-3-flash` (same board for all)
2. Tracks per-model costs in real-time
3. Generates/loads the tournament schedule
4. Projects total tournament cost based on measured costs

### Running a Tournament

```bash
# 1. Generate round-robin schedule (or use cost-estimate which auto-generates)
uv run based codenames schedule --seed 42 --output logs/eval

# 2. Run the evaluation (8 parallel threads)
uv run based codenames eval --schedule logs/eval/schedule.yml --threads 8

# 3. Retry any failed games
uv run based codenames retry --schedule logs/eval/schedule.yml
```

### Schedule Structure

For N canonical models, the schedule generates:
- **Games per matchup**: 4 (2 as red/home, 2 as blue/away)
- **Total games**: `N × (N-1) / 2 × 4`
- **Board slots**: 4 unique boards used across all matchups for difficulty analysis

### Output Files

```
logs/eval/
├── schedule.yml          # Generated tournament schedule
├── completed_games.yml   # Successfully completed games
├── failed_games.yml      # Failed games (for retry)
├── results.csv           # Bradley-Terry compatible output
└── logs/                 # Detailed game logs
```

### Bradley-Terry Ranking

The `results.csv` is compatible with [arena-rank](https://github.com/lmarena/arena-rank):

```python
import pandas as pd
from arena_rank import compute_elo

df = pd.read_csv("logs/eval/results.csv")
# columns: model_a, model_b, winner
elo_scores = compute_elo(df)
```

### CLI Feedback

During evaluation, each game completion shows:
```
✅ Game 0847/1740 | gemini-3-flash defeated gpt5-mini (red: 9/9, blue: 5/8) | $0.0234 | Total: $142.87
```

## Available Models

The framework supports 80+ models through OpenRouter, organized by capability:

**Thinking Models** (reasoning-optimized):
- `o3`, `o3-mini`, `o4-mini` - OpenAI reasoning
- `gpt5`, `gpt5.1`, `gpt5.2` - OpenAI GPT-5 family
- `grok4`, `grok3-mini` - xAI reasoning
- `gemini-2.5`, `gemini-flash` - Google reasoning
- `sonnet-4`, `opus-4`, `opus-4.5` - Anthropic reasoning
- `deepseek-r1` - DeepSeek reasoning

**Standard Models**:
- `gpt4`, `gpt4o`, `gpt4.1` - OpenAI standard
- `claude`, `sonnet-3.5`, `haiku-3.5` - Anthropic standard
- `grok3` - xAI standard
- `llama-3.3`, `qwen3`, `deepseek-v3` - Open source

## Project Structure

```
based-eval/
├── cli.py                      # Unified CLI entry point
│
├── codenames/                  # Codenames game
│   ├── cli_codenames.py        # Codenames CLI (run, schedule, eval, retry)
│   ├── game.py                 # Game logic (VERSION = "3.0.0")
│   ├── player.py               # AI/Human players
│   ├── prompt_manager.py       # Prompt templates
│   └── prompts/                # Role prompts
│       ├── red_spymaster.md
│       ├── blue_spymaster.md
│       ├── red_operative.md
│       ├── blue_operative.md
│       └── referee.md
│
├── connections/                # Connections game
│   ├── src/connections_eval/
│   │   ├── cli.py              # Connections CLI commands
│   │   └── core.py             # Game logic (VERSION = "3.0.0")
│   ├── inputs/                 # Puzzles and prompts
│   │   ├── connections_puzzles.yml
│   │   └── prompt_template.xml
│   └── scripts/                # Data processing
│       ├── extract_summaries.py   # Extract run data from MotherDuck
│       └── create_results_table_gt.py
│
├── shared/                     # Shared infrastructure
│   ├── cli_analytics.py        # Analytics CLI commands
│   ├── controllog/             # Double-entry logging SDK
│   │   ├── sdk.py              # Core event/posting system
│   │   └── builders.py         # High-level event builders
│   ├── adapters/
│   │   └── openrouter_adapter.py  # Unified OpenRouter client
│   ├── utils/
│   │   ├── retry.py            # Exponential backoff
│   │   ├── timing.py           # Performance timing
│   │   ├── tokens.py           # Token counting
│   │   ├── logging.py          # JSON logging
│   │   └── motherduck.py       # MotherDuck integration
│   └── inputs/
│       └── model_mappings.yml  # Model config + canonical_models list
│
├── inputs/                     # Codenames word bank
│   └── names.yaml
│
├── results/                    # Evaluation results
│   └── run_summaries.csv       # Aggregated run data
│
├── docs/                       # GitHub Pages (results, logs)
└── logs/                       # Game logs & analytics
    └── eval/                   # Tournament evaluation output
        ├── schedule.yml
        ├── completed_games.yml
        ├── failed_games.yml
        └── results.csv         # Bradley-Terry input
```

## Shared Infrastructure

### controllog SDK

Double-entry accounting system for structured event logging:

```python
from shared import controllog as cl

cl.init(project_id="codenames", log_dir=Path("logs"))

# Log run start with version tracking (v3.0.0+)
cl.event(
    kind="run_start",
    payload={"version": "3.0.0", "model": "gemini-flash", "seed": 42},
    run_id=run_id,
    project_id="codenames",
)

# Log model calls with balanced postings
cl.model_prompt(task_id=task_id, agent_id="spymaster", ...)
cl.model_completion(task_id=task_id, wall_ms=latency, cost_money=cost, ...)

# Track state transitions
cl.state_move(task_id=task_id, from_="WIP", to="DONE")
```

**Version Tracking**: Starting with v3.0.0, `run_start` events capture the evaluation version, enabling accurate historical analysis even as the framework evolves.

### OpenRouter Adapter

Unified API client with thinking model support:

```python
from shared.adapters import chat, OpenRouterAdapter

# Function-based API
response = chat(messages, model="google/gemini-2.5-flash")

# Class-based API
adapter = OpenRouterAdapter()
content, metadata = adapter.call_model_with_metadata("gemini-flash", prompt)
```

### MotherDuck Integration

Upload controllog to MotherDuck for analytics:

```bash
export MOTHERDUCK_DB="md:based_eval"

# Automatic upload after Codenames game completion
uv run based codenames run --red gemini-flash --blue gemini-flash

# Manual upload for all logs
uv run based analytics upload

# Run trial balance to verify double-entry accounting
uv run based analytics trial-balance
```

## Logging & Analytics

### controllog (Structured)
- `logs/controllog/YYYY-MM-DD/events.jsonl` - All events
- `logs/controllog/YYYY-MM-DD/postings.jsonl` - Balanced ledger entries

### Codenames Logs
- `logs/play_by_play_*.log` - Human-readable game events
- `logs/box_scores_*.jsonl` - Team performance summaries
- `logs/game_metadata_*.jsonl` - AI call metrics

### Connections Logs
- `logs/connections_eval_*.jsonl` - Structured game logs

## Development

### Running Tests
```bash
# Run all tests from project root
PYTHONPATH=".:connections/src" uv run pytest tests/
PYTHONPATH=".:connections/src" uv run pytest connections/tests/
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

## Unified CLI

The BASED framework provides a unified CLI for all games:

```bash
# Codenames - Single Game
uv run based codenames run --red MODEL --blue MODEL
uv run based codenames list-models
uv run based codenames prompt --role red-spymaster

# Codenames - Tournament Evaluation
uv run based codenames list-canonical              # List qualified models
uv run based codenames cost-estimate --seed 42     # Estimate tournament cost
uv run based codenames schedule --seed 42          # Generate round-robin schedule
uv run based codenames eval --schedule FILE        # Run tournament
uv run based codenames retry --schedule FILE       # Retry failed games

# Connections  
uv run based connections run --model MODEL --puzzles N
uv run based connections list-models
uv run based connections upload  # Upload logs to MotherDuck

# Analytics
uv run based analytics trial-balance  # Verify double-entry accounting
uv run based analytics upload         # Upload all logs to MotherDuck
uv run based analytics leaderboard    # Bradley-Terry rankings from tournament
```

## Latest Results

### Connections Leaderboard
[📊 View Interactive Results Table](https://matsonj.github.io/eval-connections/) - Sports-style box score showing latest model performance

*Table includes solve rates, costs, token usage, and timing metrics. Models with ≥50% win rate qualify for Codenames tournament.*

### Codenames Tournament
After running a tournament evaluation, use [arena-rank](https://github.com/lmarena/arena-rank) to compute Bradley-Terry rankings from `results.csv`.

## License

MIT License

## Credits

- Codenames game design by Vlaada Chvátil. [Official Rules](https://czechgames.com/files/rules/codenames-rules-en.pdf)
- Connections puzzle by The New York Times
