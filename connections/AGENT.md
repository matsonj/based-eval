# AGENT.md - Connections Eval Project

## Commands

### Running Evaluations
```bash
# Run canonical puzzle set (20 hardest puzzles) - RECOMMENDED for model comparison
uv run based connections run --model MODEL_NAME --canonical

# Run with parallel execution (default 8 threads)
uv run based connections run --model gpt5 --canonical --threads 8

# Run specific puzzles
uv run based connections run --model gpt5 --puzzle-ids "813,814,821"

# Run N random puzzles (legacy mode - NOT recommended for comparison)
uv run based connections run --model gpt5 --puzzles 10

# Interactive mode
uv run based connections run --interactive
```

### Puzzle Difficulty Ranking
```bash
# Rank a single puzzle (10 runs on gemini-3-flash)
uv run based connections rank --puzzle-id 813 --runs 10

# Quick test (1 run)
uv run based connections rank --puzzle-id 813 --runs 1

# Rank all puzzles in parallel
uv run based connections rank --threads 4

# List puzzles with difficulty info
uv run based connections list-puzzles --difficulty
```

### Other Commands
- **Run tests**: `uv run pytest` (all tests) or `uv run pytest tests/test_cli.py::test_specific`
- **List models**: `uv run based connections list-models`
- **Install deps**: `uv sync`
- **Extract data**: `uv run python scripts/extract_summaries.py`
- **Generate table**: `uv run python scripts/create_results_table_gt.py`

## Puzzle Distribution Fix (v3.0.0)

### The Problem
Before v3.0.0, different models received different puzzle sets due to:
1. Seed defaulting to `int(time.time())` when not specified
2. `--puzzles N` truncating a shuffled list → different models got different subsets

This made cross-model win rate comparisons **invalid**.

### The Solution
1. **Canonical puzzles**: Mark hardest puzzles with `canonical: true` in `connections_puzzles.yml`
2. **`--canonical` flag**: Run only canonical puzzles for consistent comparison
3. **`rank` command**: Empirically determine puzzle difficulty using gemini-3-flash

### How Ranking Works
1. Run each puzzle 10x on gemini-3-flash
2. Track: win rate, completion tokens, guesses, mistakes
3. Calculate difficulty score: `(win_rate * 100) - token_penalty`
4. Lower score = harder puzzle
5. Select 20 hardest as canonical set

### Tier Thresholds
- **Hard**: win_rate < 50%
- **Medium**: 50% ≤ win_rate < 80%  
- **Easy**: win_rate ≥ 80%

## Architecture
- **Core**: `src/connections_eval/core.py` - Game logic, puzzle loading, difficulty ranking
- **CLI**: `src/connections_eval/cli.py` - Typer CLI with run, rank, list-puzzles commands
- **Adapters**: Uses `shared/adapters/openrouter_adapter.py` for OpenRouter API
- **Data files**:
  - `inputs/connections_puzzles.yml` - Puzzle definitions with `canonical: true` flags
  - `inputs/puzzle_difficulty.yml` - Difficulty rankings from gemini-3-flash
  - `inputs/prompt_template.xml` - Prompt template for AI
- **Shared**: `../shared/inputs/model_mappings.yml` (unified model mappings)
- **Logs**: JSONL format in `logs/` with exchange and summary data

## Key Data Classes
```python
@dataclass
class Puzzle:
    id: int
    date: str
    difficulty: float
    words: List[str]
    groups: List[PuzzleGroup]
    canonical: bool = False  # Part of canonical eval set

@dataclass
class PuzzleDifficultyResult:
    puzzle_id: int
    win_rate: float
    avg_completion_tokens: float
    difficulty_score: float  # Lower = harder
```

## Code Style
- **Imports**: Standard library first, then third-party, then local imports
- **Types**: Use `typing` annotations with dataclasses for structured data
- **Naming**: Snake_case for functions/variables, PascalCase for classes
- **Strings**: Use f-strings for formatting, XML templates for prompts  
- **Error handling**: Retry with exponential backoff for API calls
- **CLI**: Use Typer with rich console output

## Version History
- **v3.0.0**: Added puzzle ranking, canonical sets, parallel execution, fixed distribution bug
- **v2.0.2**: Fixed reasoning field extraction for thinking models
