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
uv run based run --red gemini-flash --blue gemini-flash
```

**Game Setup:**
- 25 words in a 5×5 grid
- Red Team: 8-9 Agents | Blue Team: 8-9 Agents
- 7 Innocent Bystanders | 1 Assassin (instant loss)

**Roles:**
- **Spymaster**: Gives one-word clues with a number
- **Operative**: Guesses words based on clues
- **Referee**: Validates clue legality

### 🔗 Connections

Based on the New York Times Connections puzzle. AI models must identify four groups of four related words from a 16-word grid.

```bash
cd connections && uv run connections_eval run --model gemini-flash --puzzles 10
```

**Game Setup:**
- 16 words to sort into 4 groups of 4
- Each group has a hidden category (e.g., "Types of keys", "Words before 'dog'")
- 4 mistakes allowed before game over

## Features

- **Multi-Game Framework**: Unified infrastructure for multiple evaluation games
- **AI vs AI**: Pit different models against each other
- **Human vs AI**: Interactive mode for human players
- **200+ Models**: OpenRouter integration for broad model coverage
- **Thinking Model Support**: Automatic detection and configuration for reasoning models (o3, gpt-5, grok-4, etc.)
- **Structured Logging**: controllog SDK with double-entry accounting
- **MotherDuck Integration**: Upload results for analytics
- **Cost Tracking**: Token usage and API cost tracking per call

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
uv run based run --red gpt4 --blue claude
```

### Run Codenames (Interactive)
```bash
uv run based run --red gpt4 --blue claude --interactive red-spymaster
```

### Run Connections
```bash
cd connections
uv run connections_eval run --model gemini-flash --puzzles 5
```

### List Available Models
```bash
uv run based list-models
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
├── based/                      # Codenames game
│   ├── cli.py                  # CLI interface
│   ├── game.py                 # Game logic
│   ├── player.py               # AI/Human players
│   └── prompt_manager.py       # Prompt templates
│
├── connections/                # Connections game
│   └── src/connections_eval/
│       ├── cli.py              # CLI interface
│       └── core.py             # Game logic
│
├── shared/                     # Shared infrastructure
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
│   ├── inputs/
│   │   └── model_mappings.yml  # Unified model config
│   └── scripts/
│       ├── load_controllog_to_motherduck.py
│       └── reports_controllog.py
│
├── prompts/                    # Codenames prompts
│   ├── red_spymaster.md
│   ├── red_operative.md
│   ├── blue_spymaster.md
│   ├── blue_operative.md
│   └── referee.md
│
├── inputs/                     # Codenames word bank
│   └── names.yaml
│
└── logs/                       # Game logs & analytics
```

## Shared Infrastructure

### controllog SDK

Double-entry accounting system for structured event logging:

```python
from shared import controllog as cl

cl.init(project_id="codenames", log_dir=Path("logs"))

# Log model calls with balanced postings
cl.model_prompt(task_id=task_id, agent_id="spymaster", ...)
cl.model_completion(task_id=task_id, wall_ms=latency, cost_money=cost, ...)

# Track state transitions
cl.state_move(task_id=task_id, from_="WIP", to="DONE")
```

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
# Logs automatically uploaded after game completion (Connections)
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
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

## Roadmap

See [TODO.md](TODO.md) for planned improvements:
- [ ] Integrate controllog into Codenames
- [ ] Unified CLI (`uv run based codenames ...` / `uv run based connections ...`)
- [ ] Shared analytics dashboard

## License

MIT License

## Credits

- Codenames game design by Vlaada Chvátil. [Official Rules](https://czechgames.com/files/rules/codenames-rules-en.pdf)
- Connections puzzle by The New York Times
