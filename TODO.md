# BASED Eval - TODO

## Remaining Migration Tasks

### Integrate controllog SDK into Codenames (`based/`)

The Codenames game is still using its old custom logging system (`based/utils/logging.py`) instead of the shared controllog SDK. This needs to be updated for unified analytics across all games.

#### Current State
- Codenames uses custom logging: `log_game_start()`, `log_spymaster_clue()`, etc.
- Connections uses controllog: `cl.model_prompt()`, `cl.model_completion()`, `cl.state_move()`

#### Tasks

- [ ] **Initialize controllog in `based/game.py`**
  ```python
  from shared import controllog as cl
  cl.init(project_id="codenames", log_dir=log_path)
  ```

- [ ] **Emit model events for AI calls**
  - Spymaster clue generation → `cl.model_prompt()` + `cl.model_completion()`
  - Operative guess generation → `cl.model_prompt()` + `cl.model_completion()`
  - Referee validation → `cl.model_prompt()` + `cl.model_completion()`

- [ ] **Track state transitions per game**
  - Game start: `cl.state_move(from_="NEW", to="WIP")`
  - Game end (win): `cl.state_move(from_="WIP", to="DONE")`
  - Game end (loss/error): `cl.state_move(from_="WIP", to="FAILED")`

- [ ] **Log balanced postings for resources**
  - Token usage (prompt_tokens, completion_tokens)
  - Wall time (latency_ms)
  - Costs (openrouter_cost, upstream_cost)

- [ ] **Add MotherDuck upload to CLI**
  - Import `upload_controllog_to_motherduck`, `validate_upload`, `run_trial_balance`
  - Add `--keep-local-files` flag
  - Upload after game completion

- [ ] **Update or deprecate old logging utilities**
  - Keep `based/utils/logging.py` for backwards compatibility (play-by-play, box scores)
  - Or migrate those to controllog event payloads

#### Benefits
- Unified analytics in MotherDuck across Codenames and Connections
- Double-entry accounting for token/cost tracking
- Trial balance validation for data integrity
- Consistent event schema across all BASED eval games

---

## Unified CLI (`uv run based codenames ...` / `uv run based connections ...`)

Create a single entry point for all BASED eval games with consistent interface.

### Current State
- Codenames: `uv run based run --red gemini-flash --blue gemini-flash`
- Connections: `uv run connections_eval run --model gemini-flash`
- Two separate CLIs with different interfaces

### Target State
```bash
# Codenames
uv run based codenames run --red gemini-flash --blue gemini-flash
uv run based codenames run --interactive

# Connections  
uv run based connections run --model gemini-flash --puzzles 10
uv run based connections run --interactive

# Shared analytics
uv run based analytics trial-balance
uv run based analytics cost-report
uv run based analytics leaderboard
```

### Tasks

- [ ] **Restructure `based/cli.py` to use subcommands**
  ```python
  app = typer.Typer()
  codenames_app = typer.Typer()
  connections_app = typer.Typer()
  analytics_app = typer.Typer()
  
  app.add_typer(codenames_app, name="codenames")
  app.add_typer(connections_app, name="connections")
  app.add_typer(analytics_app, name="analytics")
  ```

- [ ] **Move Codenames CLI commands under `codenames` subcommand**
  - `based codenames run` - Run Codenames game(s)
  - `based codenames list-models` - List available models

- [ ] **Integrate Connections CLI under `connections` subcommand**
  - `based connections run` - Run Connections evaluation
  - `based connections list-models` - List available models
  - Import from `connections.src.connections_eval.cli`

- [ ] **Create shared `analytics` subcommand**
  - `based analytics trial-balance` - Run trial balance check on MotherDuck
  - `based analytics cost-report` - Show cost breakdown by model/game
  - `based analytics leaderboard` - Show model performance across games

- [ ] **Update `pyproject.toml` entry point**
  ```toml
  [project.scripts]
  based = "based.cli:app"
  ```

- [ ] **Shared CLI options**
  - `--log-path` - Directory for logs (default: `logs/`)
  - `--verbose` - Enable verbose output
  - `--motherduck-db` - MotherDuck connection string
  - `--keep-local-files` - Keep local controllog files after upload

- [ ] **Deprecate standalone `connections_eval` CLI**
  - Keep for backwards compatibility temporarily
  - Add deprecation warning pointing to `based connections`

### File Structure After Migration
```
based/
├── cli.py              # Main CLI with subcommands
├── cli_codenames.py    # Codenames-specific commands
├── cli_connections.py  # Connections-specific commands (wrapper)
├── cli_analytics.py    # Shared analytics commands
└── ...
```

---

## Future Enhancements

### Additional Games
- [ ] Framework for adding new games to BASED eval
- [ ] Shared prompt template system
- [ ] Game-agnostic leaderboard/results dashboard

