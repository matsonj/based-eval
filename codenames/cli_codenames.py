"""CLI subcommand for Codenames game."""

import csv
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from codenames.game import CodenamesGame
from codenames.player import AIPlayer, HumanPlayer, Player
from codenames.prompt_manager import PromptManager
from codenames.utils.logging import setup_logging
from shared.utils.motherduck import (
    upload_controllog_to_motherduck,
    validate_upload,
    run_trial_balance,
    cleanup_local_files,
)
from shared.adapters.openrouter_adapter import _load_model_mappings as _shared_load_model_mappings
from shared import controllog as cl

app = typer.Typer(help="Run Codenames games for AI evaluation")
console = Console()


def _load_model_mappings(mappings_file: Optional[str] = None) -> dict:
    """Load model mappings from shared infrastructure."""
    try:
        # Use the shared model mappings loader (doesn't require API key)
        mappings = _shared_load_model_mappings()
        # Flatten the hierarchical structure
        flat = {}
        if "thinking" in mappings:
            flat.update(mappings["thinking"])
        if "non_thinking" in mappings:
            flat.update(mappings["non_thinking"])
        return flat
    except Exception:
        # Fallback to basic mappings
        return {
            "gpt4": "openai/gpt-4",
            "claude": "anthropic/claude-3.5-sonnet",
            "gemini": "google/gemini-2.5-pro",
        }


def _load_canonical_models() -> List[str]:
    """Load the list of canonical models for full evaluation."""
    try:
        mappings_file = Path(__file__).parent.parent / "shared" / "inputs" / "model_mappings.yml"
        with open(mappings_file, 'r') as f:
            data = yaml.safe_load(f)
        return data.get("canonical_models", [])
    except Exception as e:
        console.print(f"[red]Error loading canonical models: {e}[/red]")
        return []


def _load_schedule(schedule_path: Path) -> Dict[str, Any]:
    """Load schedule from YAML file."""
    with open(schedule_path, 'r') as f:
        return yaml.safe_load(f)


def _save_schedule(schedule: Dict[str, Any], schedule_path: Path) -> None:
    """Save schedule to YAML file."""
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schedule_path, 'w') as f:
        yaml.dump(schedule, f, default_flow_style=False, sort_keys=False)


def _load_completed_games(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load completed games from YAML file."""
    completed_path = eval_dir / "completed_games.yml"
    if completed_path.exists():
        with open(completed_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            return {g["game_id"]: g for g in data.get("games", [])}
    return {}


def _save_completed_game(eval_dir: Path, game_result: Dict[str, Any]) -> None:
    """Append a completed game to the YAML file."""
    completed_path = eval_dir / "completed_games.yml"
    
    # Load existing
    if completed_path.exists():
        with open(completed_path, 'r') as f:
            data = yaml.safe_load(f) or {"games": []}
    else:
        data = {"games": []}
    
    data["games"].append(game_result)
    
    with open(completed_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _load_failed_games(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load failed games from YAML file."""
    failed_path = eval_dir / "failed_games.yml"
    if failed_path.exists():
        with open(failed_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            return {g["game_id"]: g for g in data.get("games", [])}
    return {}


def _save_failed_game(eval_dir: Path, game_failure: Dict[str, Any]) -> None:
    """Append a failed game to the YAML file."""
    failed_path = eval_dir / "failed_games.yml"
    
    # Load existing
    if failed_path.exists():
        with open(failed_path, 'r') as f:
            data = yaml.safe_load(f) or {"games": []}
    else:
        data = {"games": []}
    
    data["games"].append(game_failure)
    
    with open(failed_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _generate_schedule(
    models: List[str],
    seed: int,
    games_per_matchup: int = 4,
) -> Dict[str, Any]:
    """Generate a round-robin tournament schedule.
    
    Each pair of models plays `games_per_matchup` games (default 4),
    alternating home (red) and away (blue) assignments.
    
    Returns a schedule dict with metadata and game list.
    """
    rng = random.Random(seed)
    
    # Generate all matchups (sorted pairs for consistency)
    models_sorted = sorted(models)
    matchups = []
    for i, model_a in enumerate(models_sorted):
        for model_b in models_sorted[i + 1:]:
            matchups.append((model_a, model_b))
    
    # Generate games for each matchup with alternating home/away
    games = []
    game_idx = 0
    for model_a, model_b in matchups:
        for slot in range(1, games_per_matchup + 1):
            # Alternate: odd slots = A is home, even slots = B is home
            if slot % 2 == 1:
                red_model, blue_model = model_a, model_b
            else:
                red_model, blue_model = model_b, model_a
            
            games.append({
                "game_id": f"game_{game_idx:04d}",
                "board_slot": slot,
                "red_model": red_model,
                "blue_model": blue_model,
                "matchup": f"{model_a}_vs_{model_b}",
            })
            game_idx += 1
    
    # Interleave games for better parallelism
    # Group by board_slot first, then shuffle within each slot
    games_by_slot = {i: [] for i in range(1, games_per_matchup + 1)}
    for game in games:
        games_by_slot[game["board_slot"]].append(game)
    
    # Shuffle within each slot for variety
    for slot_games in games_by_slot.values():
        rng.shuffle(slot_games)
    
    # Interleave: take one from each slot in round-robin fashion
    interleaved_games = []
    max_games_per_slot = max(len(g) for g in games_by_slot.values())
    for i in range(max_games_per_slot):
        for slot in range(1, games_per_matchup + 1):
            if i < len(games_by_slot[slot]):
                interleaved_games.append(games_by_slot[slot][i])
    
    # Reassign game_ids after interleaving
    for idx, game in enumerate(interleaved_games):
        game["game_id"] = f"game_{idx:04d}"
    
    return {
        "metadata": {
            "seed": seed,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_games": len(interleaved_games),
            "total_matchups": len(matchups),
            "games_per_matchup": games_per_matchup,
            "models": models_sorted,
        },
        "games": interleaved_games,
    }


def _validate_api_keys_and_models(red: Optional[str], blue: Optional[str], referee: Optional[str], interactive: bool):
    """Validate that required API keys are present and model names are valid."""
    # Check API keys
    missing_keys = []
    
    if not os.getenv("OPENROUTER_API_KEY"):
        missing_keys.append("OPENROUTER_API_KEY")
    
    if missing_keys:
        console.print(f"[red]Error: Missing required environment variable(s): {', '.join(missing_keys)}[/red]")
        console.print("[yellow]You may need to `source .env` if you are running locally[/yellow]")
        console.print("[yellow]Or set the environment variable directly:[/yellow]")
        for key in missing_keys:
            console.print(f"[yellow]  export {key}='your-key-here'[/yellow]")
        raise typer.Exit(1)
    
    # Check model names
    model_mappings = _load_model_mappings()
    available_models = list(model_mappings.keys())
    invalid_models = []
    
    # Check models that will be used
    models_to_check = []
    if red:
        models_to_check.append(("red", red))
    if blue:
        models_to_check.append(("blue", blue))
    if referee:
        models_to_check.append(("referee", referee))
    
    for team, model in models_to_check:
        if model not in available_models:
            invalid_models.append((team, model))
    
    if invalid_models:
        console.print(f"[red]Error: Invalid model name(s):[/red]")
        for team, model in invalid_models:
            console.print(f"[red]  {team}: '{model}'[/red]")
        console.print(f"\n[yellow]Available models:[/yellow]")
        for model in sorted(available_models):
            console.print(f"[yellow]  {model}[/yellow]")
        console.print(f"\n[yellow]Use 'uv run based codenames list-models' for detailed model information[/yellow]")
        raise typer.Exit(1)


def _format_board_for_operative_cli(board_state: dict) -> str:
    """Format the board for operative display with revealed status."""
    board = board_state["board"]
    revealed = board_state["revealed"]
    
    # Create a 5x5 grid display
    lines = []
    for row in range(5):
        row_items = []
        for col in range(5):
            idx = row * 5 + col
            word = board[idx]
            
            # Mark revealed words with brackets
            if revealed.get(word, False):
                display_word = f"[{word}]"
            else:
                display_word = word
            
            row_items.append(f"{display_word:>12}")
        
        lines.append(" |".join(row_items))
    
    return "\n".join(lines)


@app.command()
def run(
    red: Optional[str] = typer.Option(None, help="Model for Red Team"),
    blue: Optional[str] = typer.Option(None, help="Model for Blue Team"),
    referee: Optional[str] = typer.Option("gemini-3-flash", help="Model for Referee (clue validation)"),
    no_referee: bool = typer.Option(False, help="Disable referee validation"),
    interactive: Optional[str] = typer.Option(
        None, help="Interactive mode: referee, red-spymaster, red-operative, blue-spymaster, blue-operative"
    ),
    num_games: int = typer.Option(1, help="Number of games to play"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducible games"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    red_spymaster_prompt: str = typer.Option(
        "codenames/prompts/red_spymaster.md", help="Red spymaster prompt file"
    ),
    red_operative_prompt: str = typer.Option(
        "codenames/prompts/red_operative.md", help="Red operative prompt file"
    ),
    blue_spymaster_prompt: str = typer.Option(
        "codenames/prompts/blue_spymaster.md", help="Blue spymaster prompt file"
    ),
    blue_operative_prompt: str = typer.Option(
        "codenames/prompts/blue_operative.md", help="Blue operative prompt file"
    ),
    referee_prompt: str = typer.Option(
        "codenames/prompts/referee.md", help="Referee prompt file"
    ),
    log_path: str = typer.Option("logs", help="Directory for log files"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
    keep_local_files: bool = typer.Option(
        False, help="Keep local controllog files after uploading to MotherDuck"
    ),
):
    """Run a Codenames game (part of BASED eval)."""

    # Validate API keys and model names first
    _validate_api_keys_and_models(red, blue, referee, interactive)

    # Setup logging
    log_dir = Path(log_path)
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir, verbose)

    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    # Validate interactive mode options
    valid_interactive_modes = ["referee", "red-spymaster", "red-operative", "blue-spymaster", "blue-operative"]
    if interactive and interactive not in valid_interactive_modes:
        console.print(
            f"[red]Error: Interactive mode must be one of: {', '.join(valid_interactive_modes)}[/red]"
        )
        raise typer.Exit(1)

    # Validate arguments
    if not interactive and (not red or not blue):
        console.print(
            "[red]Error: Must specify both --red and --blue models, or use --interactive mode[/red]"
        )
        raise typer.Exit(1)

    if interactive and interactive != "referee" and (not red or not blue):
        console.print(
            "[red]Error: Interactive modes other than 'referee' require both --red and --blue models[/red]"
        )
        raise typer.Exit(1)

    # Create players
    try:
        red_player: Player
        blue_player: Player

        if interactive:
            if interactive == "referee":
                # Referee mode: both teams are AI, human is referee
                if not red or not blue:
                    console.print("[red]Error: Referee mode requires both --red and --blue models[/red]")
                    raise typer.Exit(1)
                red_player = AIPlayer(red)
                blue_player = AIPlayer(blue)
            else:
                # Role-specific mode: specific role is human, rest are AI
                red_player = AIPlayer(red)
                blue_player = AIPlayer(blue)
                console.print(f"[green]Interactive mode: Human {interactive}[/green]")
        else:
            if not red or not blue:
                console.print(
                    "[red]Error: Non-interactive mode requires both --red and --blue models[/red]"
                )
                raise typer.Exit(1)
            red_player = AIPlayer(red)
            blue_player = AIPlayer(blue)

        # Create referee player if not disabled
        referee_player = None
        if not no_referee:
            if interactive == "referee":
                referee_player = HumanPlayer()
            elif referee:
                try:
                    referee_player = AIPlayer(referee)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not create referee player: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error creating players: {e}[/red]")
        raise typer.Exit(1)

    # Generate run_id for controllog tracking
    run_id = f"{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}_codenames_{red or 'human'}_{blue or 'human'}"

    # Run games
    results = []
    game_ids = []
    for game_num in range(num_games):
        console.print(f"\n[bold]Game {game_num + 1}/{num_games}[/bold]")

        try:
            game = CodenamesGame(
                words_file=words_file,
                red_player=red_player,
                blue_player=blue_player,
                referee_player=referee_player,
                red_spymaster_prompt=red_spymaster_prompt,
                red_operative_prompt=red_operative_prompt,
                blue_spymaster_prompt=blue_spymaster_prompt,
                blue_operative_prompt=blue_operative_prompt,
                referee_prompt=referee_prompt,
                interactive_mode=interactive,
            )

            # Initialize controllog for this game
            game.init_controllog(log_dir, run_id)
            game_ids.append(game.game_id)

            result = game.play()
            results.append(result)

            # Display game result
            console.print(f"[bold]Game {game_num + 1} Result:[/bold]")
            if result["winner"]:
                console.print(f"[green]Winner: {result['winner'].title()} Team[/green]")
            else:
                console.print("[yellow]Game ended in a draw[/yellow]")

        except Exception as e:
            logger.error(f"Error in game {game_num + 1}: {e}")
            console.print(f"[red]Error in game {game_num + 1}: {e}[/red]")

    # Display summary
    if len(results) > 1:
        display_summary(results)

    # Upload to MotherDuck if configured
    motherduck_db = os.getenv("MOTHERDUCK_DB")
    if motherduck_db and results:
        console.print()
        console.print("📤 Uploading controllog to MotherDuck...", style="bold blue")

        # Upload
        upload_success = upload_controllog_to_motherduck(log_dir, motherduck_db)
        if upload_success:
            console.print("✅ Upload successful", style="green")

            # Validate upload
            console.print("🔍 Validating upload...", style="dim")
            validation_success = validate_upload(run_id, motherduck_db)
            if validation_success:
                console.print("✅ Validation passed", style="green")
            else:
                console.print("⚠️  Validation failed: run_id not found in database", style="yellow")

            # Run trial balance
            console.print("⚖️  Running trial balance check...", style="dim")
            trial_balance_success = run_trial_balance(motherduck_db)
            if trial_balance_success:
                console.print("✅ Trial balance passed", style="green")
            else:
                console.print("⚠️  Trial balance check failed", style="yellow")

            # Cleanup local files if not keeping them
            if not keep_local_files:
                console.print("🧹 Cleaning up local files...", style="dim")
                cleanup_local_files(log_dir, run_id, keep_local_files)
                console.print("✅ Local files cleaned up", style="green")
            else:
                console.print("📁 Keeping local files (--keep-local-files flag set)", style="dim")
        else:
            console.print("❌ Upload failed", style="red")
            console.print("⚠️  Local files retained due to upload failure", style="yellow")


def display_summary(results: list):
    """Display summary statistics for multiple games."""
    total_games = len(results)
    red_wins = sum(1 for r in results if r.get("winner") == "red")
    blue_wins = sum(1 for r in results if r.get("winner") == "blue")
    draws = total_games - red_wins - blue_wins

    table = Table(title="Game Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Games", str(total_games))
    table.add_row("Red Team Wins", str(red_wins))
    table.add_row("Blue Team Wins", str(blue_wins))
    table.add_row("Draws", str(draws))
    table.add_row("Red Win Rate", f"{red_wins/total_games*100:.1f}%")
    table.add_row("Blue Win Rate", f"{blue_wins/total_games*100:.1f}%")

    console.print(table)


@app.command()
def list_models():
    """List available AI models for Codenames."""
    try:
        # Load model mappings directly (doesn't require API key)
        model_mappings = _load_model_mappings()

        # Create a nice table
        table = Table(title="Available AI Models")
        table.add_column("CLI Name", style="cyan", min_width=15)
        table.add_column("OpenRouter Model ID", style="magenta", min_width=30)
        table.add_column("Provider", style="green", min_width=12)

        # Sort models by name
        sorted_models = sorted(model_mappings.keys())

        for model_name in sorted_models:
            model_id = model_mappings[model_name]
            provider = model_id.split("/")[0] if "/" in model_id else "Unknown"
            table.add_row(model_name, model_id, provider)

        console.print(table)
        console.print(f"\n✨ Total: {len(model_mappings)} models available")
        console.print(
            "\n💡 Usage: [bold]uv run based codenames run --red [model] --blue [model][/bold]"
        )

    except Exception as e:
        console.print(f"[red]Error loading models: {e}[/red]")
        console.print(
            "Make sure the model mappings file exists at shared/inputs/model_mappings.yml"
        )


@app.command()
def prompt(
    role: str = typer.Argument(..., help="Role to test: spymaster, operative, or referee"),
    team: str = typer.Option("red", help="Team color: red or blue"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducible board generation"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    clue: str = typer.Option("EXAMPLE", help="Sample clue for operative/referee prompts"),
    number: str = typer.Option("2", help="Sample number for operative/referee prompts (can be 'unlimited' or '0')"),
    red_spymaster_prompt: str = typer.Option("codenames/prompts/red_spymaster.md", help="Red spymaster prompt file"),
    red_operative_prompt: str = typer.Option("codenames/prompts/red_operative.md", help="Red operative prompt file"),
    blue_spymaster_prompt: str = typer.Option("codenames/prompts/blue_spymaster.md", help="Blue spymaster prompt file"),
    blue_operative_prompt: str = typer.Option("codenames/prompts/blue_operative.md", help="Blue operative prompt file"),
    referee_prompt: str = typer.Option("codenames/prompts/referee.md", help="Referee prompt file"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Test and display the exact prompt sent to AI agents."""
    
    # Setup logging (but don't create game logs for prompt testing)
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    setup_logging(temp_dir, verbose)
    
    # Validate role
    valid_roles = ["spymaster", "operative", "referee"]
    if role not in valid_roles:
        console.print(f"[red]Error: Role must be one of: {', '.join(valid_roles)}[/red]")
        raise typer.Exit(1)
    
    # Validate team
    if team not in ["red", "blue"]:
        console.print(f"[red]Error: Team must be 'red' or 'blue'[/red]")
        raise typer.Exit(1)
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        console.print(f"[dim]Using seed: {seed}[/dim]")
    
    try:
        # Create a game to generate realistic board state
        red_player = HumanPlayer()  # Dummy players
        blue_player = HumanPlayer()
        
        game = CodenamesGame(
            words_file=words_file,
            red_player=red_player,
            blue_player=blue_player,
            red_spymaster_prompt=red_spymaster_prompt,
            red_operative_prompt=red_operative_prompt,
            blue_spymaster_prompt=blue_spymaster_prompt,
            blue_operative_prompt=blue_operative_prompt,
            referee_prompt=referee_prompt,
        )
        
        # Setup the board
        game.setup_board()
        
        # Get board state 
        board_state = game.get_board_state(reveal_all=(role == "spymaster"))
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        
        # Parse number parameter (handle unlimited and 0)
        parsed_number: int|str
        if number.lower() == "unlimited":
            parsed_number = "unlimited"
        elif number == "0":
            parsed_number = 0
        else:
            try:
                parsed_number = int(number)
            except ValueError:
                console.print(f"[red]Error: Number must be an integer, 'unlimited', or '0'[/red]")
                raise typer.Exit(1)
        
        # Generate the appropriate prompt
        if role == "spymaster":
            prompt_file = red_spymaster_prompt if team == "red" else blue_spymaster_prompt
            
            # Calculate remaining agents for spymaster context
            red_remaining = sum(
                1 for word, identity in board_state["identities"].items()
                if identity == "red_agent" and not board_state["revealed"].get(word, False)
            )
            blue_remaining = sum(
                1 for word, identity in board_state["identities"].items()
                if identity == "blue_agent" and not board_state["revealed"].get(word, False)
            )
            revealed_words = [word for word, revealed in board_state["revealed"].items() if revealed]
            
            # Categorize identities for cleaner prompt formatting
            red_agents = [word for word, identity in board_state["identities"].items() 
                             if identity == "red_agent"]
            blue_agents = [word for word, identity in board_state["identities"].items() 
                              if identity == "blue_agent"]
            bystanders = [word for word, identity in board_state["identities"].items() 
                        if identity == "bystander"]
            assassin = [word for word, identity in board_state["identities"].items() 
                   if identity == "assassin"]
            
            prompt_text = prompt_manager.load_prompt(
                prompt_file,
                {
                    "board": board_state["board"],
                    "revealed": ", ".join(revealed_words) if revealed_words else "None",
                    "team": team,
                    "red_remaining": red_remaining,
                    "blue_remaining": blue_remaining,
                    "red_agents": ", ".join(red_agents),
                    "blue_agents": ", ".join(blue_agents),
                    "bystanders": ", ".join(bystanders),
                    "assassin": ", ".join(assassin),
                    "clue_history": "No previous clues yet",
                },
            )
            
        elif role == "operative":
            prompt_file = red_operative_prompt if team == "red" else blue_operative_prompt
            
            # Filter board to only show available (unrevealed) words for operative
            available_words = [
                word for word in board_state["board"] 
                if not board_state["revealed"].get(word, False)
            ]
            
            # Format available words as a simple list
            available_words_formatted = ", ".join(available_words)
            
            prompt_text = prompt_manager.load_prompt(
                prompt_file,
                {
                    "board": _format_board_for_operative_cli(board_state),
                    "available_words": available_words_formatted,
                    "clue_history": board_state.get("clue_history", "None (game just started)"),
                    "clue": clue,
                    "number": parsed_number,
                    "team": team,
                },
            )
            
        elif role == "referee":
            # Get team's agents for referee context
            team_agents = [
                word for word, identity in board_state["identities"].items()
                if identity == f"{team}_agent"
            ]
            
            prompt_text = prompt_manager.load_prompt(
                referee_prompt,
                {
                    "clue": clue,
                    "number": parsed_number,
                    "team": team,
                    "board": ", ".join(board_state["board"]),
                    "team_agents": ", ".join(team_agents),
                },
            )
        
        # Display the results
        console.print(f"\n[bold]🎯 {role.title()} Prompt for {team.title()} Team[/bold]")
        console.print(f"[dim]Seed: {seed}, Board: {len(board_state['board'])} words[/dim]")
        
        if role in ["operative", "referee"]:
            console.print(f"[dim]Sample clue: '{clue}' ({parsed_number})[/dim]")
        
        console.print(f"\n[yellow]{'='*80}[/yellow]")
        console.print("[yellow]PROMPT CONTENT:[/yellow]")
        console.print(f"[yellow]{'='*80}[/yellow]\n")
        
        console.print(prompt_text)
        
        console.print(f"\n[yellow]{'='*80}[/yellow]")
        console.print(f"[green]✅ Prompt generated successfully ({len(prompt_text)} characters)[/green]")
        
        # Show board state for context
        if role == "spymaster":
            console.print(f"\n[bold]📋 Board State (Spymaster View - All Identities Revealed):[/bold]")
            game.display_board(reveal_all=True)
        else:
            console.print(f"\n[bold]📋 Board State (Public View):[/bold]")
            game.display_board(reveal_all=False)
        
    except Exception as e:
        console.print(f"[red]Error generating prompt: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def schedule(
    seed: int = typer.Option(42, help="Random seed for reproducible board generation"),
    output: Path = typer.Option(
        Path("logs/eval"),
        "--output", "-o",
        help="Output directory for schedule and results"
    ),
    games_per_matchup: int = typer.Option(4, help="Number of games per model pair (default 4)"),
):
    """Generate a round-robin tournament schedule for canonical models.
    
    Creates a schedule.yml file with all games to be played. Each pair of 
    canonical models plays the specified number of games, alternating home
    (red) and away (blue) assignments for balance.
    """
    # Load canonical models
    canonical_models = _load_canonical_models()
    
    if not canonical_models:
        console.print("[red]Error: No canonical models found in model_mappings.yml[/red]")
        raise typer.Exit(1)
    
    # Validate all canonical models exist in mappings
    model_mappings = _load_model_mappings()
    invalid_models = [m for m in canonical_models if m not in model_mappings]
    if invalid_models:
        console.print(f"[red]Error: Invalid canonical models: {invalid_models}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]🎯 Generating Codenames Tournament Schedule[/bold blue]")
    console.print(f"Models: {len(canonical_models)} canonical models")
    console.print(f"Games per matchup: {games_per_matchup}")
    console.print(f"Seed: {seed}")
    
    # Generate schedule
    schedule_data = _generate_schedule(canonical_models, seed, games_per_matchup)
    
    # Save schedule
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "schedule.yml"
    _save_schedule(schedule_data, schedule_path)
    
    # Print summary
    total_games = schedule_data["metadata"]["total_games"]
    total_matchups = schedule_data["metadata"]["total_matchups"]
    
    console.print(f"\n[green]✅ Schedule generated successfully![/green]")
    console.print(f"Total matchups: {total_matchups}")
    console.print(f"Total games: {total_games}")
    console.print(f"Schedule saved to: {schedule_path}")
    
    # Show per-model breakdown
    games_per_model = (len(canonical_models) - 1) * games_per_matchup
    home_games = games_per_model // 2
    away_games = games_per_model // 2
    
    console.print(f"\nPer-model breakdown:")
    console.print(f"  Total games: {games_per_model}")
    console.print(f"  Home (red): {home_games}")
    console.print(f"  Away (blue): {away_games}")
    
    console.print(f"\n💡 Run evaluation with: [bold]uv run based codenames eval --schedule {schedule_path}[/bold]")


def _run_single_game(
    game_spec: Dict[str, Any],
    seed: int,
    words_file: str,
    log_dir: Path,
    run_id: str,
    prompt_files: Dict[str, str],
    lock: threading.Lock,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Run a single game from the schedule.
    
    Returns: (game_id, result_dict, error_message)
    """
    game_id = game_spec["game_id"]
    board_slot = game_spec["board_slot"]
    red_model = game_spec["red_model"]
    blue_model = game_spec["blue_model"]
    
    try:
        # Create players
        red_player = AIPlayer(red_model)
        blue_player = AIPlayer(blue_model)
        referee_player = AIPlayer("gemini-3-flash")
        
        # Create game with deterministic seed based on board_slot
        game = CodenamesGame(
            words_file=words_file,
            red_player=red_player,
            blue_player=blue_player,
            referee_player=referee_player,
            **prompt_files,
        )
        
        # Set deterministic board based on seed + board_slot
        # This ensures all games in the same slot use the same board
        board_seed = seed + board_slot
        random.seed(board_seed)
        
        # Initialize controllog
        game.init_controllog(log_dir, run_id)
        
        # Play the game
        result = game.play()
        
        # Extract costs from the game
        # Note: Cost tracking would need to be added to the game class
        # For now, we'll estimate based on metadata
        game_cost = 0.0
        if hasattr(game, 'total_cost'):
            game_cost = game.total_cost
        
        # Determine winner for Bradley-Terry format
        winner = result.get("winner")
        if winner == "red":
            bt_winner = "model_a"  # model_a is always red
        elif winner == "blue":
            bt_winner = "model_b"  # model_b is always blue
        else:
            bt_winner = "tie"
        
        # Calculate agent counts
        red_found = sum(
            1 for word, identity in game.identities.items()
            if identity == "red_agent" and game.revealed.get(word, False)
        )
        blue_found = sum(
            1 for word, identity in game.identities.items()
            if identity == "blue_agent" and game.revealed.get(word, False)
        )
        
        # Get total counts
        red_total = sum(1 for i in game.identities.values() if i == "red_agent")
        blue_total = sum(1 for i in game.identities.values() if i == "blue_agent")
        
        return game_id, {
            "game_id": game_id,
            "board_slot": board_slot,
            "red_model": red_model,
            "blue_model": blue_model,
            "winner": winner,
            "bt_winner": bt_winner,
            "red_agents_found": red_found,
            "red_agents_total": red_total,
            "blue_agents_found": blue_found,
            "blue_agents_total": blue_total,
            "turns": result.get("turns", 0),
            "duration_sec": result.get("duration", 0),
            "cost": game_cost,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "win_type": "assassin" if result.get("winner") and result.get("turns", 0) < 10 else "agents",
        }, None
        
    except Exception as e:
        return game_id, None, str(e)


@app.command("eval")
def run_eval(
    schedule_file: Path = typer.Option(
        ...,
        "--schedule", "-s",
        help="Path to schedule.yml file"
    ),
    threads: int = typer.Option(8, "--threads", "-t", help="Number of parallel threads"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    red_spymaster_prompt: str = typer.Option(
        "codenames/prompts/red_spymaster.md", help="Red spymaster prompt file"
    ),
    red_operative_prompt: str = typer.Option(
        "codenames/prompts/red_operative.md", help="Red operative prompt file"
    ),
    blue_spymaster_prompt: str = typer.Option(
        "codenames/prompts/blue_spymaster.md", help="Blue spymaster prompt file"
    ),
    blue_operative_prompt: str = typer.Option(
        "codenames/prompts/blue_operative.md", help="Blue operative prompt file"
    ),
    referee_prompt: str = typer.Option(
        "codenames/prompts/referee.md", help="Referee prompt file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run the full tournament evaluation from a schedule file.
    
    Runs all games defined in the schedule with parallel execution.
    Progress is tracked in completed_games.yml and failed_games.yml,
    allowing for resumption if interrupted.
    """
    # Validate API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    
    # Load schedule
    if not schedule_file.exists():
        console.print(f"[red]Error: Schedule file not found: {schedule_file}[/red]")
        raise typer.Exit(1)
    
    schedule_data = _load_schedule(schedule_file)
    eval_dir = schedule_file.parent
    
    # Setup logging
    log_dir = eval_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir, verbose)
    
    # Load completed and failed games
    completed_games = _load_completed_games(eval_dir)
    failed_games = _load_failed_games(eval_dir)
    
    # Filter out already completed games
    all_games = schedule_data["games"]
    remaining_games = [g for g in all_games if g["game_id"] not in completed_games]
    
    total_games = len(all_games)
    completed_count = len(completed_games)
    remaining_count = len(remaining_games)
    
    console.print(f"[bold blue]🎯 Codenames Tournament Evaluation[/bold blue]")
    console.print(f"Schedule: {schedule_file}")
    console.print(f"Total games: {total_games}")
    console.print(f"Already completed: {completed_count}")
    console.print(f"Remaining: {remaining_count}")
    console.print(f"Threads: {threads}")
    
    if remaining_count == 0:
        console.print("[green]✅ All games already completed![/green]")
        _generate_results_csv(eval_dir)
        return
    
    # Prepare prompt files dict
    prompt_files = {
        "red_spymaster_prompt": red_spymaster_prompt,
        "red_operative_prompt": red_operative_prompt,
        "blue_spymaster_prompt": blue_spymaster_prompt,
        "blue_operative_prompt": blue_operative_prompt,
        "referee_prompt": referee_prompt,
    }
    
    # Track running totals
    total_cost = sum(g.get("cost", 0) for g in completed_games.values())
    new_completed = 0
    new_failed = 0
    lock = threading.Lock()
    
    # Generate run_id
    run_id = f"eval_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}"
    seed = schedule_data["metadata"]["seed"]
    
    # Initialize controllog and emit run_start event
    try:
        cl.init(project_id="codenames", log_dir=log_dir)
        cl.event(
            kind="run_start",
            actor={"agent_id": "agent:codenames"},
            run_id=run_id,
            payload={
                "version": CodenamesGame.VERSION,
                "seed": seed,
                "threads": threads,
                "total_games": total_games,
                "models": schedule_data["metadata"]["models"],
            },
            project_id="codenames",
            source="runtime",
        )
    except Exception:
        pass  # Don't fail if telemetry init fails
    
    console.print(f"\n[bold]Starting evaluation...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running games...", total=remaining_count)
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(
                    _run_single_game,
                    game_spec,
                    seed,
                    words_file,
                    log_dir,
                    run_id,
                    prompt_files,
                    lock,
                ): game_spec
                for game_spec in remaining_games
            }
            
            for future in as_completed(futures):
                game_spec = futures[future]
                game_id, result, error = future.result()
                
                game_num = completed_count + new_completed + new_failed + 1
                
                if result:
                    # Success
                    with lock:
                        _save_completed_game(eval_dir, result)
                        total_cost += result.get("cost", 0)
                        new_completed += 1
                    
                    # Format result message
                    winner = result["winner"]
                    red_model = result["red_model"]
                    blue_model = result["blue_model"]
                    red_found = result["red_agents_found"]
                    red_total = result["red_agents_total"]
                    blue_found = result["blue_agents_found"]
                    blue_total = result["blue_agents_total"]
                    game_cost = result.get("cost", 0)
                    
                    if winner == "red":
                        msg = f"[green]✅ Game {game_num:04d}/{total_games} | {red_model} defeated {blue_model} (red: {red_found}/{red_total}, blue: {blue_found}/{blue_total}) | ${game_cost:.4f} | Total: ${total_cost:.2f}[/green]"
                    elif winner == "blue":
                        msg = f"[green]✅ Game {game_num:04d}/{total_games} | {blue_model} defeated {red_model} (red: {red_found}/{red_total}, blue: {blue_found}/{blue_total}) | ${game_cost:.4f} | Total: ${total_cost:.2f}[/green]"
                    else:
                        msg = f"[yellow]⚠️ Game {game_num:04d}/{total_games} | {red_model} vs {blue_model} - No winner | ${game_cost:.4f} | Total: ${total_cost:.2f}[/yellow]"
                    
                    progress.console.print(msg)
                else:
                    # Failure
                    with lock:
                        _save_failed_game(eval_dir, {
                            "game_id": game_id,
                            "board_slot": game_spec["board_slot"],
                            "red_model": game_spec["red_model"],
                            "blue_model": game_spec["blue_model"],
                            "error": error,
                            "failed_at": datetime.utcnow().isoformat() + "Z",
                        })
                        new_failed += 1
                    
                    msg = f"[red]❌ Game {game_num:04d}/{total_games} | FAILED: {error} | {game_spec['red_model']} vs {game_spec['blue_model']}[/red]"
                    progress.console.print(msg)
                
                progress.advance(task)
    
    # Summary
    console.print(f"\n[bold]Evaluation Complete![/bold]")
    console.print(f"Completed: {new_completed}")
    console.print(f"Failed: {new_failed}")
    console.print(f"Total cost: ${total_cost:.2f}")
    
    # Generate results CSV for Bradley-Terry
    _generate_results_csv(eval_dir)
    
    if new_failed > 0:
        console.print(f"\n💡 Retry failed games with: [bold]uv run based codenames retry --schedule {schedule_file}[/bold]")


def _generate_results_csv(eval_dir: Path) -> None:
    """Generate Bradley-Terry compatible CSV from completed games."""
    completed_games = _load_completed_games(eval_dir)
    
    if not completed_games:
        console.print("[yellow]No completed games to generate results from[/yellow]")
        return
    
    csv_path = eval_dir / "results.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model_a", "model_b", "winner"])
        
        for game in completed_games.values():
            # model_a is always red, model_b is always blue
            model_a = game["red_model"]
            model_b = game["blue_model"]
            winner = game.get("bt_winner", "tie")
            
            writer.writerow([model_a, model_b, winner])
    
    console.print(f"[green]📊 Results CSV saved to: {csv_path}[/green]")
    console.print("[dim]Use with arena-rank for Bradley-Terry analysis[/dim]")


@app.command()
def retry(
    schedule_file: Path = typer.Option(
        ...,
        "--schedule", "-s",
        help="Path to schedule.yml file"
    ),
    threads: int = typer.Option(8, "--threads", "-t", help="Number of parallel threads"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    red_spymaster_prompt: str = typer.Option(
        "codenames/prompts/red_spymaster.md", help="Red spymaster prompt file"
    ),
    red_operative_prompt: str = typer.Option(
        "codenames/prompts/red_operative.md", help="Red operative prompt file"
    ),
    blue_spymaster_prompt: str = typer.Option(
        "codenames/prompts/blue_spymaster.md", help="Blue spymaster prompt file"
    ),
    blue_operative_prompt: str = typer.Option(
        "codenames/prompts/blue_operative.md", help="Blue operative prompt file"
    ),
    referee_prompt: str = typer.Option(
        "codenames/prompts/referee.md", help="Referee prompt file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Retry failed games from a previous evaluation run.
    
    Reads failed_games.yml and attempts to re-run those games.
    Successfully completed games are moved to completed_games.yml.
    """
    # Validate API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    
    # Load schedule and failed games
    if not schedule_file.exists():
        console.print(f"[red]Error: Schedule file not found: {schedule_file}[/red]")
        raise typer.Exit(1)
    
    schedule_data = _load_schedule(schedule_file)
    eval_dir = schedule_file.parent
    
    failed_games = _load_failed_games(eval_dir)
    
    if not failed_games:
        console.print("[green]✅ No failed games to retry![/green]")
        return
    
    console.print(f"[bold blue]🔄 Retrying Failed Games[/bold blue]")
    console.print(f"Failed games to retry: {len(failed_games)}")
    
    # Setup logging
    log_dir = eval_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir, verbose)
    
    # Build game specs from failed games
    failed_game_specs = []
    for game_id, failure in failed_games.items():
        failed_game_specs.append({
            "game_id": game_id,
            "board_slot": failure["board_slot"],
            "red_model": failure["red_model"],
            "blue_model": failure["blue_model"],
        })
    
    # Prepare prompt files dict
    prompt_files = {
        "red_spymaster_prompt": red_spymaster_prompt,
        "red_operative_prompt": red_operative_prompt,
        "blue_spymaster_prompt": blue_spymaster_prompt,
        "blue_operative_prompt": blue_operative_prompt,
        "referee_prompt": referee_prompt,
    }
    
    # Track results
    completed_games = _load_completed_games(eval_dir)
    total_cost = sum(g.get("cost", 0) for g in completed_games.values())
    new_completed = 0
    still_failed = 0
    lock = threading.Lock()
    
    run_id = f"retry_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}"
    seed = schedule_data["metadata"]["seed"]
    
    # Clear failed games file (we'll re-add any that still fail)
    failed_path = eval_dir / "failed_games.yml"
    if failed_path.exists():
        failed_path.unlink()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Retrying games...", total=len(failed_game_specs))
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(
                    _run_single_game,
                    game_spec,
                    seed,
                    words_file,
                    log_dir,
                    run_id,
                    prompt_files,
                    lock,
                ): game_spec
                for game_spec in failed_game_specs
            }
            
            for future in as_completed(futures):
                game_spec = futures[future]
                game_id, result, error = future.result()
                
                if result:
                    with lock:
                        _save_completed_game(eval_dir, result)
                        total_cost += result.get("cost", 0)
                        new_completed += 1
                    
                    winner = result["winner"]
                    red_model = result["red_model"]
                    blue_model = result["blue_model"]
                    
                    if winner == "red":
                        msg = f"[green]✅ {red_model} defeated {blue_model}[/green]"
                    elif winner == "blue":
                        msg = f"[green]✅ {blue_model} defeated {red_model}[/green]"
                    else:
                        msg = f"[yellow]⚠️ {red_model} vs {blue_model} - No winner[/yellow]"
                    
                    progress.console.print(msg)
                else:
                    with lock:
                        _save_failed_game(eval_dir, {
                            "game_id": game_id,
                            "board_slot": game_spec["board_slot"],
                            "red_model": game_spec["red_model"],
                            "blue_model": game_spec["blue_model"],
                            "error": error,
                            "failed_at": datetime.utcnow().isoformat() + "Z",
                        })
                        still_failed += 1
                    
                    msg = f"[red]❌ FAILED again: {error} | {game_spec['red_model']} vs {game_spec['blue_model']}[/red]"
                    progress.console.print(msg)
                
                progress.advance(task)
    
    console.print(f"\n[bold]Retry Complete![/bold]")
    console.print(f"Recovered: {new_completed}")
    console.print(f"Still failing: {still_failed}")
    console.print(f"Total cost: ${total_cost:.2f}")
    
    # Regenerate results CSV
    _generate_results_csv(eval_dir)
    
    if still_failed > 0:
        console.print(f"\n[yellow]⚠️ {still_failed} games still failing. Check failed_games.yml for details.[/yellow]")


@app.command()
def list_canonical():
    """List canonical models for tournament evaluation."""
    canonical_models = _load_canonical_models()
    model_mappings = _load_model_mappings()
    
    if not canonical_models:
        console.print("[yellow]No canonical models defined in model_mappings.yml[/yellow]")
        return
    
    table = Table(title="Canonical Models for Tournament")
    table.add_column("#", style="dim", justify="right")
    table.add_column("CLI Name", style="cyan")
    table.add_column("OpenRouter Model ID", style="magenta")
    table.add_column("Provider", style="green")
    
    for i, model_name in enumerate(canonical_models, 1):
        model_id = model_mappings.get(model_name, "UNKNOWN")
        provider = model_id.split("/")[0] if "/" in model_id else "Unknown"
        
        if model_id == "UNKNOWN":
            table.add_row(str(i), model_name, "[red]NOT FOUND[/red]", "")
        else:
            table.add_row(str(i), model_name, model_id, provider)
    
    console.print(table)
    console.print(f"\n✨ Total: {len(canonical_models)} canonical models")
    
    # Calculate tournament stats
    n = len(canonical_models)
    matchups = n * (n - 1) // 2
    games_4 = matchups * 4
    
    console.print(f"\n📊 Tournament stats (4 games per matchup):")
    console.print(f"   Unique matchups: {matchups}")
    console.print(f"   Total games: {games_4}")
    console.print(f"   Games per model: {(n - 1) * 4}")


def _run_cost_estimation_game(
    model: str,
    seed: int,
    words_file: str,
    log_dir: Path,
    run_id: str,
    prompt_files: Dict[str, str],
    lock: threading.Lock,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Run a single cost estimation game: model vs gemini-3-flash.
    
    Returns: (model_name, result_dict, error_message)
    """
    try:
        # Create players - model is red (home), gemini-3-flash is blue (away)
        red_player = AIPlayer(model)
        blue_player = AIPlayer("gemini-3-flash")
        referee_player = AIPlayer("gemini-3-flash")
        
        # Create game with deterministic seed
        game = CodenamesGame(
            words_file=words_file,
            red_player=red_player,
            blue_player=blue_player,
            referee_player=referee_player,
            **prompt_files,
        )
        
        # Set deterministic board based on seed
        random.seed(seed)
        
        # Initialize controllog
        game.init_controllog(log_dir, run_id)
        
        # Play the game
        result = game.play()
        
        # Extract costs from the game
        game_cost = getattr(game, 'total_cost', 0.0)
        upstream_cost = getattr(game, 'total_upstream_cost', 0.0)
        
        # Determine winner
        winner = result.get("winner")
        model_won = (winner == "red")  # model is always red
        
        # Calculate agent counts
        red_found = sum(
            1 for word, identity in game.identities.items()
            if identity == "red_agent" and game.revealed.get(word, False)
        )
        blue_found = sum(
            1 for word, identity in game.identities.items()
            if identity == "blue_agent" and game.revealed.get(word, False)
        )
        
        red_total = sum(1 for i in game.identities.values() if i == "red_agent")
        blue_total = sum(1 for i in game.identities.values() if i == "blue_agent")
        
        return model, {
            "model": model,
            "opponent": "gemini-3-flash",
            "winner": winner,
            "model_won": model_won,
            "red_agents_found": red_found,
            "red_agents_total": red_total,
            "blue_agents_found": blue_found,
            "blue_agents_total": blue_total,
            "turns": result.get("turns", 0),
            "duration_sec": result.get("duration", 0),
            "cost": game_cost,
            "upstream_cost": upstream_cost,
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }, None
        
    except Exception as e:
        return model, None, str(e)


@app.command("cost-estimate")
def cost_estimate(
    seed: int = typer.Option(42, help="Random seed for reproducible board generation"),
    threads: int = typer.Option(8, "--threads", "-t", help="Number of parallel threads"),
    output: Path = typer.Option(
        Path("logs/eval"),
        "--output", "-o",
        help="Output directory for schedule and results"
    ),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    red_spymaster_prompt: str = typer.Option(
        "codenames/prompts/red_spymaster.md", help="Red spymaster prompt file"
    ),
    red_operative_prompt: str = typer.Option(
        "codenames/prompts/red_operative.md", help="Red operative prompt file"
    ),
    blue_spymaster_prompt: str = typer.Option(
        "codenames/prompts/blue_spymaster.md", help="Blue spymaster prompt file"
    ),
    blue_operative_prompt: str = typer.Option(
        "codenames/prompts/blue_operative.md", help="Blue operative prompt file"
    ),
    referee_prompt: str = typer.Option(
        "codenames/prompts/referee.md", help="Referee prompt file"
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
    games_per_matchup: int = typer.Option(4, help="Number of games per model pair for schedule"),
):
    """Estimate tournament cost by running each canonical model vs gemini-3-flash.
    
    Runs one game per canonical model against gemini-3-flash, all using the same
    board (seed) for consistency. After completion, generates/loads the tournament
    schedule and projects the total cost based on measured per-model costs.
    """
    # Load canonical models
    canonical_models = _load_canonical_models()
    
    if not canonical_models:
        console.print("[red]Error: No canonical models found in model_mappings.yml[/red]")
        raise typer.Exit(1)
    
    # Setup directories
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "cost_estimate_logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir, verbose)
    
    # Generate run_id for this estimation
    run_id = f"cost_estimate_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}"
    
    # Initialize controllog
    try:
        cl.init(project_id="codenames", log_dir=log_dir)
        cl.event(
            kind="run_start",
            actor={"agent_id": "agent:codenames"},
            run_id=run_id,
            payload={
                "version": CodenamesGame.VERSION,
                "seed": seed,
                "threads": threads,
                "total_games": len(canonical_models),
                "estimation_mode": True,
            },
            project_id="codenames",
            source="runtime",
        )
    except Exception:
        pass
    
    console.print(f"[bold blue]💰 Codenames Tournament Cost Estimation[/bold blue]")
    console.print(f"Models to test: {len(canonical_models)} canonical models")
    console.print(f"Each model plays 1 game vs gemini-3-flash (same board)")
    console.print(f"Seed: {seed}")
    console.print(f"Threads: {threads}")
    
    # Prepare prompt files
    prompt_files = {
        "red_spymaster_prompt": red_spymaster_prompt,
        "red_operative_prompt": red_operative_prompt,
        "blue_spymaster_prompt": blue_spymaster_prompt,
        "blue_operative_prompt": blue_operative_prompt,
        "referee_prompt": referee_prompt,
    }
    
    # Track results
    results: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = {}
    total_cost = 0.0
    completed_count = 0
    lock = threading.Lock()
    
    console.print(f"\n[bold]Starting cost estimation games...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Running games...", total=len(canonical_models)
        )
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all games
            futures = {}
            for model in canonical_models:
                future = executor.submit(
                    _run_cost_estimation_game,
                    model,
                    seed,
                    words_file,
                    log_dir,
                    run_id,
                    prompt_files,
                    lock,
                )
                futures[future] = model
            
            # Process results as they complete
            for future in as_completed(futures):
                model = futures[future]
                model_name, result, error = future.result()
                
                with lock:
                    completed_count += 1
                    
                    if error:
                        failed[model_name] = error
                        console.print(f"[red]❌ {model_name}: {error}[/red]")
                    else:
                        results[model_name] = result
                        game_cost = result.get("cost", 0)
                        total_cost += game_cost
                        
                        winner = result["winner"]
                        model_won = result["model_won"]
                        status = "🏆" if model_won else "😢"
                        
                        console.print(
                            f"{status} Game {completed_count}/{len(canonical_models)} | "
                            f"[cyan]{model_name}[/cyan] vs gemini-3-flash | "
                            f"Winner: [{'green' if model_won else 'yellow'}]{winner}[/] | "
                            f"Score: ({result['red_agents_found']}/{result['red_agents_total']} vs "
                            f"{result['blue_agents_found']}/{result['blue_agents_total']}) | "
                            f"[green]${game_cost:.4f}[/green] | "
                            f"Total: [bold green]${total_cost:.4f}[/bold green]"
                        )
                    
                    progress.update(task, advance=1)
    
    # Summary table
    console.print(f"\n[bold blue]📊 Cost Estimation Results[/bold blue]\n")
    
    table = Table(title="Per-Model Costs (vs gemini-3-flash)")
    table.add_column("Model", style="cyan")
    table.add_column("Result", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Turns", justify="right")
    table.add_column("Cost", style="green", justify="right")
    
    sorted_models = sorted(results.keys(), key=lambda m: results[m].get("cost", 0), reverse=True)
    
    for model in sorted_models:
        r = results[model]
        result_str = "Win" if r["model_won"] else "Loss"
        score_str = f"{r['red_agents_found']}/{r['red_agents_total']} - {r['blue_agents_found']}/{r['blue_agents_total']}"
        table.add_row(
            model,
            result_str,
            score_str,
            str(r.get("turns", 0)),
            f"${r.get('cost', 0):.4f}",
        )
    
    # Add failed models to table
    for model, error in failed.items():
        table.add_row(model, "[red]FAILED[/red]", "-", "-", "[red]N/A[/red]")
    
    console.print(table)
    
    # Calculate statistics
    if results:
        costs = [r.get("cost", 0) for r in results.values()]
        avg_cost = sum(costs) / len(costs)
        max_cost = max(costs)
        min_cost = min(costs)
        
        console.print(f"\n[bold]Cost Statistics:[/bold]")
        console.print(f"  Total estimation cost: ${total_cost:.4f}")
        console.print(f"  Average cost per game: ${avg_cost:.4f}")
        console.print(f"  Min cost per game: ${min_cost:.4f}")
        console.print(f"  Max cost per game: ${max_cost:.4f}")
    
    # Save estimation results
    estimation_results_path = output / "cost_estimation_results.yml"
    estimation_data = {
        "metadata": {
            "run_id": run_id,
            "seed": seed,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_cost": total_cost,
            "games_run": len(results),
            "games_failed": len(failed),
        },
        "per_model_costs": {model: r.get("cost", 0) for model, r in results.items()},
        "results": list(results.values()),
        "failed": failed,
    }
    
    with open(estimation_results_path, "w") as f:
        yaml.dump(estimation_data, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"\n[green]✅ Estimation results saved to: {estimation_results_path}[/green]")
    
    # Now project tournament cost
    console.print(f"\n[bold blue]📈 Tournament Cost Projection[/bold blue]\n")
    
    # Check for existing schedule or generate one
    schedule_path = output / "schedule.yml"
    if schedule_path.exists():
        console.print(f"[dim]Loading existing schedule from {schedule_path}[/dim]")
        schedule_data = _load_schedule(schedule_path)
    else:
        console.print(f"[dim]Generating new schedule (games_per_matchup={games_per_matchup})...[/dim]")
        schedule_data = _generate_schedule(canonical_models, seed, games_per_matchup)
        _save_schedule(schedule_data, schedule_path)
        console.print(f"[green]Schedule saved to: {schedule_path}[/green]")
    
    total_games = schedule_data["metadata"]["total_games"]
    
    # Build a cost model based on estimation results
    # For each game in the schedule, estimate cost as:
    # (model_a_cost + model_b_cost) / 2 * scaling_factor
    # where scaling_factor accounts for games being more complex than estimation
    
    # Calculate per-model average cost from estimation
    model_costs = {model: r.get("cost", 0) for model, r in results.items()}
    
    # Add gemini-3-flash cost (it was the opponent in all games, so estimate from any game)
    # Rough approximation: assume gemini-3-flash is about 30% of total game cost
    if model_costs:
        avg_total_cost = sum(model_costs.values()) / len(model_costs)
        gemini_flash_estimated_cost = avg_total_cost * 0.3  # Conservative estimate
        model_costs["gemini-3-flash"] = gemini_flash_estimated_cost
    
    # Project costs for each game in schedule
    projected_costs = []
    missing_models = set()
    
    for game in schedule_data["games"]:
        red_model = game["red_model"]
        blue_model = game["blue_model"]
        
        red_cost = model_costs.get(red_model, 0)
        blue_cost = model_costs.get(blue_model, 0)
        
        if red_model not in model_costs:
            missing_models.add(red_model)
        if blue_model not in model_costs:
            missing_models.add(blue_model)
        
        # Game cost is sum of both players' estimated costs
        # Scale by 1.5 to account for longer games in tournament vs estimation
        game_cost = (red_cost + blue_cost) * 1.2
        projected_costs.append(game_cost)
    
    if missing_models:
        console.print(f"[yellow]⚠️ Missing cost data for models: {missing_models}[/yellow]")
        console.print("[yellow]   Using $0 for missing models (underestimate)[/yellow]")
    
    projected_total = sum(projected_costs)
    projected_avg = projected_total / len(projected_costs) if projected_costs else 0
    
    # Build projection table by model
    model_game_counts: Dict[str, int] = {}
    model_projected_costs: Dict[str, float] = {}
    
    for game in schedule_data["games"]:
        red_model = game["red_model"]
        blue_model = game["blue_model"]
        
        model_game_counts[red_model] = model_game_counts.get(red_model, 0) + 1
        model_game_counts[blue_model] = model_game_counts.get(blue_model, 0) + 1
        
        red_cost = model_costs.get(red_model, 0) * 1.2
        blue_cost = model_costs.get(blue_model, 0) * 1.2
        
        model_projected_costs[red_model] = model_projected_costs.get(red_model, 0) + red_cost
        model_projected_costs[blue_model] = model_projected_costs.get(blue_model, 0) + blue_cost
    
    # Show top 10 most expensive models
    sorted_by_cost = sorted(model_projected_costs.keys(), key=lambda m: model_projected_costs[m], reverse=True)
    
    proj_table = Table(title="Projected Cost by Model (Top 10)")
    proj_table.add_column("Model", style="cyan")
    proj_table.add_column("Games", justify="right")
    proj_table.add_column("Est. Cost/Game", style="yellow", justify="right")
    proj_table.add_column("Projected Total", style="green", justify="right")
    proj_table.add_column("% of Budget", justify="right")
    
    for model in sorted_by_cost[:10]:
        games = model_game_counts[model]
        per_game = model_costs.get(model, 0)
        total_model_cost = model_projected_costs[model]
        pct = (total_model_cost / projected_total * 100) if projected_total > 0 else 0
        
        proj_table.add_row(
            model,
            str(games),
            f"${per_game:.4f}",
            f"${total_model_cost:.2f}",
            f"{pct:.1f}%",
        )
    
    console.print(proj_table)
    
    # Final summary
    console.print(f"\n[bold]Tournament Cost Projection Summary:[/bold]")
    console.print(f"  Total games in schedule: {total_games}")
    console.print(f"  Average projected cost per game: ${projected_avg:.4f}")
    console.print(f"  [bold green]Projected total cost: ${projected_total:.2f}[/bold green]")
    
    # Show cost ranges
    low_estimate = projected_total * 0.7
    high_estimate = projected_total * 1.5
    console.print(f"\n[dim]Cost range (accounting for variance):[/dim]")
    console.print(f"  Low estimate (70%):  ${low_estimate:.2f}")
    console.print(f"  High estimate (150%): ${high_estimate:.2f}")
    
    console.print(f"\n💡 To run the tournament: [bold]uv run based codenames eval --schedule {schedule_path}[/bold]")

