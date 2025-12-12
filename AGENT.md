# Agent Instructions for BASED Eval

## Project Overview
BASED (Benchmark for Association, Sorting, and Entity Deduction) is a multi-game AI evaluation framework. Currently implements Codenames - a strategic word association game where teams use AI spymasters and operatives to identify agents while avoiding the assassin.

## Development Environment

### Key Commands
- **Install dependencies**: `uv sync`
- **Run Codenames game**: `uv run based codenames run --red gpt4 --blue claude`
- **Run Connections**: `uv run based connections run --model gpt4 --puzzles 10`
- **Interactive mode**: `uv run based codenames run --red gpt4 --blue claude --interactive red-spymaster`
- **Test prompts**: `uv run based codenames prompt spymaster --seed 42`
- **Analytics**: `uv run based analytics trial-balance`
- **Run tests**: `uv run pytest`
- **Format code**: `uv run black . && uv run isort .`
- **Type check**: `uv run mypy codenames/`

### Project Structure
```
codenames/                  # Codenames game package
├── cli.py                  # Main CLI with subcommands
├── cli_codenames.py        # Codenames-specific commands
├── cli_connections.py      # Connections wrapper commands
├── cli_analytics.py        # Shared analytics commands
├── game.py                 # Codenames game logic and board management
├── player.py               # AIPlayer and HumanPlayer classes
├── prompt_manager.py       # Markdown template loader
├── adapters/
│   └── openrouter_adapter.py  # OpenRouter API client
└── utils/
    └── logging.py          # Logging configuration

shared/                     # Shared infrastructure
├── controllog/             # Double-entry accounting SDK for logging
├── adapters/               # Shared API adapters
└── utils/                  # Common utilities (retry, timing, tokens, motherduck)

inputs/
├── names.yaml              # Word bank for game boards
└── model_mappings.yml      # Model alias configuration

prompts/                    # Markdown prompt templates
├── red_spymaster.md
├── red_operative.md
├── blue_spymaster.md
├── blue_operative.md
├── referee.md
└── shared/
    └── game_rules.md

logs/                       # Game logs and analysis data
├── based_*.log             # Detailed debug logs
├── play_by_play_*.log      # Clean game events (clues, guesses, results)
├── box_scores_*.jsonl      # Team performance summaries with formatted boards
├── game_metadata_*.jsonl   # AI call metrics (tokens, costs, latency, results)
└── referee_*.log           # Consolidated referee validation logs
```

## Code Style & Conventions
- Use **type hints** throughout
- Follow **PEP 8** formatting
- Use **rich** for console output formatting
- Use **typer** for CLI interface
- Use **pydantic** for data validation where applicable
- Log structured data to **JSONL** for analysis

## Architecture Notes

### Codenames Game Flow
1. **Board Setup**: 25 words assigned random identities (9 red, 8 blue, 7 bystanders, 1 assassin)
2. **Turn Loop**: Teams alternate Spymaster → Operative phases
3. **Spymaster Phase**: AI/Human gives one-word clue + number
4. **Operative Phase**: AI/Human makes up to N+1 guesses
5. **Win Conditions**: Find all agents OR opponent guesses assassin

### Key Design Principles
- **Stateless AI Calls**: Each OpenRouter request is independent (security requirement)
- **External Prompts**: All AI prompts loaded from Markdown files for easy tuning
- **Flexible Player Types**: Same interface for AI and Human players
- **Comprehensive Logging**: Both human-readable and structured machine-readable logs

### Testing Strategy
- Unit tests for game logic (board setup, identity assignment, win conditions)
- Integration tests for AI player interactions
- Mock OpenRouter API calls for reliable testing
- Test both AI vs AI and Human vs AI modes

## Environment Variables
- `OPENROUTER_API_KEY`: Required for AI model access

## Common Development Tasks

### Testing AI Prompts
- **Test spymaster prompts**: `uv run based codenames prompt spymaster --seed 42 --team red`
- **Test operative prompts**: `uv run based codenames prompt operative --seed 42 --clue "TOOLS" --number 3`
- **Test referee prompts**: `uv run based codenames prompt referee --seed 42 --clue "WEAPONS" --number 2`
- **Test expert clues**: `uv run based codenames prompt operative --clue "ANIMALS" --number 0` or `--number unlimited`

### Adding New AI Models
1. Update `shared/inputs/model_mappings.yml`
2. Test with `uv run based codenames run --red NEW_MODEL --blue claude`

### Modifying Game Rules
- Edit board setup logic in `game.py`
- Update win/lose conditions in `process_guess()`
- Adjust N+1 rule implementation in operative guess handling

### Customizing AI Prompts
- Edit Markdown files in `prompts/` directory
- Use template variables: `{{BOARD}}`, `{{CLUE}}`, etc.
- Test changes with `--verbose` flag to see full AI exchanges

### Debugging AI Behavior
1. Use `--verbose` to see all AI exchanges
2. Check `logs/` directory for detailed JSONL data
3. Modify prompt templates to adjust AI strategy
4. Test with `--seed` for reproducible games

### Game Analysis & Performance Tracking
The framework produces multiple log formats for different analysis needs:

1. **Play-by-Play Logs** (`play_by_play_*.log`)
   - Clean, human-readable game events
   - Shows board state, clues, guesses, and results

2. **Box Score Data** (`box_scores_*.jsonl`)
   - Team performance summaries in JSONL format
   - Includes accuracy, move counts, win/loss data

3. **Game Metadata** (`game_metadata_*.jsonl`)
   - Detailed AI call metrics (tokens, costs, latency)
   - Turn-by-turn results and success/failure tracking

4. **Referee Logs** (`referee_*.log`)
   - Consolidated clue validation logs with team headers
   - Shows all referee decisions and reasoning

## Terminology Reference

| Codenames Term | Description |
|---------------|-------------|
| Spymaster | Gives clues to guide operatives |
| Operative | Makes guesses based on clues |
| Agent | Team member to find (Red/Blue Agent) |
| Bystander | Innocent neutral word |
| Assassin | Instant loss if guessed |
| Clue | One-word hint + number |
| Guess | Operative's word selection |

## Known Issues & TODOs
- [x] Implement model listing command (`list-models`) ✅
- [x] Add configuration file support for model mappings ✅
- [x] Implement expert clue types (0 and unlimited) ✅
- [x] Implement comprehensive logging with cost tracking ✅
- [ ] Add more games to BASED eval framework
- [ ] Improve AI response parsing robustness
- [ ] Add game replay functionality from JSONL logs
- [ ] Implement tournament mode for model evaluation

## Dependencies
- **typer**: CLI framework
- **rich**: Terminal formatting
- **openai**: OpenRouter API client
- **pyyaml**: Configuration files
- **tenacity**: Retry logic for API calls
- **pydantic**: Data validation
