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

## Future Enhancements

### Unified CLI
- [ ] Create top-level `based` CLI with subcommands:
  - `based codenames run ...`
  - `based connections run ...`
  - `based analytics ...` (shared reports)

### Additional Games
- [ ] Framework for adding new games to BASED eval
- [ ] Shared prompt template system
- [ ] Game-agnostic leaderboard/results dashboard

