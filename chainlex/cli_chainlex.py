"""CLI subcommand for ChainLex-1 game."""

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

from chainlex.game import ChainLexGame
from chainlex.player import AIPlayer
from codenames.utils.logging import setup_logging
from shared.adapters.openrouter_adapter import _load_model_mappings as _shared_load_model_mappings
from shared import controllog as cl

app = typer.Typer(help="Run ChainLex-1 games for AI evaluation")
console = Console()


def _load_model_mappings() -> dict:
    """Load model mappings from shared infrastructure."""
    try:
        mappings = _shared_load_model_mappings()
        flat = {}
        if "thinking" in mappings:
            flat.update(mappings["thinking"])
        if "non_thinking" in mappings:
            flat.update(mappings["non_thinking"])
        return flat
    except Exception:
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


def _validate_api_keys_and_models(model_a: Optional[str], model_b: Optional[str]):
    """Validate that required API keys are present and model names are valid."""
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
        console.print("[yellow]Try `source .env` if running locally[/yellow]")
        raise typer.Exit(1)
    
    model_mappings = _load_model_mappings()
    available_models = list(model_mappings.keys())
    invalid_models = []
    
    if model_a and model_a not in available_models:
        invalid_models.append(("model-a", model_a))
    if model_b and model_b not in available_models:
        invalid_models.append(("model-b", model_b))
    
    if invalid_models:
        console.print("[red]Error: Invalid model name(s):[/red]")
        for role, model_name in invalid_models:
            console.print(f"[red]  {role}: '{model_name}'[/red]")
        console.print(f"\n[yellow]Available models:[/yellow]")
        for m in sorted(available_models)[:10]:
            console.print(f"[yellow]  {m}[/yellow]")
        console.print("[yellow]  ...[/yellow]")
        console.print("\n[yellow]Use 'uv run based chainlex list-models' for full list[/yellow]")
        raise typer.Exit(1)


@app.command()
def run(
    model_a: str = typer.Option(..., "--model-a", "-a", help="First model (Player A)"),
    model_b: str = typer.Option(..., "--model-b", "-b", help="Second model (Player B)"),
    num_games: int = typer.Option(1, help="Number of games to play"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducible games"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    clue_giver_prompt: str = typer.Option(
        "chainlex/prompts/clue_giver.md", help="Clue giver prompt file"
    ),
    guesser_prompt: str = typer.Option(
        "chainlex/prompts/guesser.md", help="Guesser prompt file"
    ),
    log_path: str = typer.Option("logs/chainlex", help="Directory for log files"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Run a ChainLex-1 head-to-head game between two models."""
    
    _validate_api_keys_and_models(model_a, model_b)

    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose)

    logger = logging.getLogger(__name__)

    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    try:
        player_a = AIPlayer(model_a)
        player_b = AIPlayer(model_b)
    except Exception as e:
        console.print(f"[red]Error creating players: {e}[/red]")
        raise typer.Exit(1)

    run_id = f"{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}_chainlex_{model_a}_vs_{model_b}"

    results = []
    for game_num in range(num_games):
        console.print(f"\n[bold]Game {game_num + 1}/{num_games}[/bold]")

        try:
            game = ChainLexGame(
                words_file=words_file,
                player_a=player_a,
                player_b=player_b,
                clue_giver_prompt=clue_giver_prompt,
                guesser_prompt=guesser_prompt,
                seed=seed + game_num if seed is not None else None,
            )

            game.init_controllog(log_dir, run_id)
            result = game.play()
            results.append(result)

        except Exception as e:
            logger.error(f"Error in game {game_num + 1}: {e}")
            console.print(f"[red]Error in game {game_num + 1}: {e}[/red]")
            import traceback
            traceback.print_exc()

    if len(results) > 1:
        _display_head_to_head_summary(results, model_a, model_b)


def _display_head_to_head_summary(results: list, model_a: str, model_b: str):
    """Display summary statistics for multiple head-to-head games."""
    total_games = len(results)
    
    wins_a = sum(1 for r in results if r.get("winner") == "model_a")
    wins_b = sum(1 for r in results if r.get("winner") == "model_b")
    ties = sum(1 for r in results if r.get("winner") == "tie")
    
    total_score_a = sum(r.get("score_a", 0) for r in results)
    total_score_b = sum(r.get("score_b", 0) for r in results)
    
    avg_score_a = total_score_a / total_games
    avg_score_b = total_score_b / total_games

    table = Table(title="ChainLex-1 Head-to-Head Summary")
    table.add_column("Metric", style="cyan")
    table.add_column(model_a, style="green")
    table.add_column(model_b, style="magenta")

    table.add_row("Wins", str(wins_a), str(wins_b))
    table.add_row("Ties", str(ties), str(ties))
    table.add_row("Win Rate", f"{wins_a/total_games*100:.1f}%", f"{wins_b/total_games*100:.1f}%")
    table.add_row("Total Score", str(total_score_a), str(total_score_b))
    table.add_row("Avg Score", f"{avg_score_a:.1f}", f"{avg_score_b:.1f}")

    console.print(table)
    
    # Declare overall winner
    if wins_a > wins_b:
        console.print(f"\n[bold green]🏆 Overall Winner: {model_a} ({wins_a}-{wins_b})[/bold green]")
    elif wins_b > wins_a:
        console.print(f"\n[bold magenta]🏆 Overall Winner: {model_b} ({wins_b}-{wins_a})[/bold magenta]")
    else:
        console.print(f"\n[bold yellow]🤝 Series Tied: {wins_a}-{wins_b}[/bold yellow]")


@app.command()
def list_models():
    """List available AI models for ChainLex-1."""
    try:
        model_mappings = _load_model_mappings()

        table = Table(title="Available AI Models")
        table.add_column("CLI Name", style="cyan", min_width=15)
        table.add_column("OpenRouter Model ID", style="magenta", min_width=30)
        table.add_column("Provider", style="green", min_width=12)

        sorted_models = sorted(model_mappings.keys())

        for model_name in sorted_models:
            model_id = model_mappings[model_name]
            provider = model_id.split("/")[0] if "/" in model_id else "Unknown"
            table.add_row(model_name, model_id, provider)

        console.print(table)
        console.print(f"\n✨ Total: {len(model_mappings)} models available")
        console.print("\n💡 Usage: [bold]uv run based chainlex run --model [model][/bold]")

    except Exception as e:
        console.print(f"[red]Error loading models: {e}[/red]")


@app.command()
def prompt(
    role: str = typer.Argument(..., help="Role to test: clue_giver or guesser"),
    seed: Optional[int] = typer.Option(42, help="Random seed for reproducible board generation"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    clue: str = typer.Option("EXAMPLE", help="Sample clue for guesser prompts"),
    number: int = typer.Option(3, help="Sample number for guesser prompts"),
    clue_giver_prompt: str = typer.Option("chainlex/prompts/clue_giver.md", help="Clue giver prompt file"),
    guesser_prompt: str = typer.Option("chainlex/prompts/guesser.md", help="Guesser prompt file"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Test and display the exact prompt sent to AI agents."""
    import tempfile
    from codenames.prompt_manager import PromptManager
    
    temp_dir = Path(tempfile.mkdtemp())
    setup_logging(temp_dir, verbose)
    
    valid_roles = ["clue_giver", "guesser"]
    if role not in valid_roles:
        console.print(f"[red]Error: Role must be one of: {', '.join(valid_roles)}[/red]")
        raise typer.Exit(1)
    
    if seed is not None:
        random.seed(seed)
        console.print(f"[dim]Using seed: {seed}[/dim]")
    
    try:
        # Create a game to generate realistic board state
        from chainlex.player import AIPlayer
        
        # Dummy player just for board setup
        class DummyPlayer:
            model_name = "dummy"
        
        game = ChainLexGame(
            words_file=words_file,
            player_a=DummyPlayer(),
            player_b=DummyPlayer(),
            clue_giver_prompt=clue_giver_prompt,
            guesser_prompt=guesser_prompt,
            seed=seed,
        )
        
        game.setup_board()
        board_state = game.get_board_state(reveal_all=(role == "clue_giver"))
        
        prompt_manager = PromptManager()
        
        if role == "clue_giver":
            friendly_words = [w for w, i in board_state["identities"].items() if i == "friendly"]
            bystanders = [w for w, i in board_state["identities"].items() if i == "bystander"]
            assassin = [w for w, i in board_state["identities"].items() if i == "assassin"]
            
            prompt_text = prompt_manager.load_prompt(
                clue_giver_prompt,
                {
                    "board": board_state["board"],
                    "friendly_words": ", ".join(friendly_words),
                    "bystanders": ", ".join(bystanders),
                    "assassin": ", ".join(assassin),
                    "num_friendly": len(friendly_words),
                },
            )
            
        elif role == "guesser":
            available_words = [w for w in board_state["board"] if not board_state["revealed"].get(w, False)]
            
            # Format board as 4x4 grid
            lines = []
            for row in range(4):
                row_items = []
                for col in range(4):
                    idx = row * 4 + col
                    word = board_state["board"][idx]
                    row_items.append(f"{word:>12}")
                lines.append(" |".join(row_items))
            board_formatted = "\n".join(lines)
            
            prompt_text = prompt_manager.load_prompt(
                guesser_prompt,
                {
                    "board": board_formatted,
                    "available_words": ", ".join(available_words),
                    "clue": clue,
                    "number": number,
                },
            )
        
        console.print(f"\n[bold]🎯 {role.replace('_', ' ').title()} Prompt[/bold]")
        console.print(f"[dim]Seed: {seed}, Board: {len(board_state['board'])} words[/dim]")
        
        if role == "guesser":
            console.print(f"[dim]Sample clue: '{clue}' ({number})[/dim]")
        
        console.print(f"\n[yellow]{'='*80}[/yellow]")
        console.print("[yellow]PROMPT CONTENT:[/yellow]")
        console.print(f"[yellow]{'='*80}[/yellow]\n")
        
        console.print(prompt_text)
        
        console.print(f"\n[yellow]{'='*80}[/yellow]")
        console.print(f"[green]✅ Prompt generated successfully ({len(prompt_text)} characters)[/green]")
        
        # Show board state
        console.print(f"\n[bold]📋 Board State:[/bold]")
        game.display_board(reveal_all=True)
        
    except Exception as e:
        console.print(f"[red]Error generating prompt: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def _run_single_game(
    model_a: str,
    model_b: str,
    seed: int,
    words_file: str,
    log_dir: Path,
    run_id: str,
    prompt_files: Dict[str, str],
    lock: threading.Lock,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Run a single ChainLex-1 head-to-head game.
    
    Returns: (game_id, result_dict, error_message)
    """
    try:
        player_a = AIPlayer(model_a)
        player_b = AIPlayer(model_b)
        
        game = ChainLexGame(
            words_file=words_file,
            player_a=player_a,
            player_b=player_b,
            quiet=True,
            seed=seed,
            **prompt_files,
        )
        
        game.init_controllog(log_dir, run_id)
        result = game.play()
        
        return game.game_id, result, None
        
    except Exception as e:
        import uuid
        return str(uuid.uuid4())[:8], None, str(e)


@app.command("eval")
def run_eval(
    models: Optional[List[str]] = typer.Option(None, "--model", "-m", help="Models to evaluate (can specify multiple)"),
    all_canonical: bool = typer.Option(False, "--all", "-a", help="Evaluate all canonical models"),
    games_per_matchup: int = typer.Option(5, "--games", "-g", help="Number of games per matchup"),
    threads: int = typer.Option(8, "--threads", "-t", help="Number of parallel threads"),
    seed: int = typer.Option(42, "--seed", help="Base random seed"),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    output: Path = typer.Option(Path("logs/chainlex/eval"), "--output", "-o", help="Output directory"),
    clue_giver_prompt: str = typer.Option("chainlex/prompts/clue_giver.md"),
    guesser_prompt: str = typer.Option("chainlex/prompts/guesser.md"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run ChainLex-1 round-robin evaluation across multiple models.
    
    Each pair of models plays head-to-head on the same board.
    Results are saved in Bradley-Terry compatible format for ranking.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY not set. Try `source .env`[/red]")
        raise typer.Exit(1)
    
    # Determine which models to evaluate
    if all_canonical:
        eval_models = _load_canonical_models()
        if not eval_models:
            console.print("[red]Error: No canonical models found[/red]")
            raise typer.Exit(1)
    elif models:
        eval_models = list(models)
    else:
        console.print("[red]Error: Specify --model or use --all for canonical models[/red]")
        raise typer.Exit(1)
    
    if len(eval_models) < 2:
        console.print("[red]Error: Need at least 2 models for head-to-head evaluation[/red]")
        raise typer.Exit(1)
    
    # Validate models
    model_mappings = _load_model_mappings()
    invalid = [m for m in eval_models if m not in model_mappings]
    if invalid:
        console.print(f"[red]Invalid models: {invalid}[/red]")
        raise typer.Exit(1)
    
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir, verbose)
    
    # Generate all matchups
    matchups = []
    eval_models_sorted = sorted(eval_models)
    for i, model_a in enumerate(eval_models_sorted):
        for model_b in eval_models_sorted[i + 1:]:
            for game_idx in range(games_per_matchup):
                matchups.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "seed": seed + game_idx,
                    "game_idx": game_idx,
                })
    
    total_games = len(matchups)
    num_matchups = len(eval_models) * (len(eval_models) - 1) // 2
    
    console.print(f"[bold blue]🎯 ChainLex-1 Round-Robin Evaluation[/bold blue]")
    console.print(f"Models: {len(eval_models)}")
    console.print(f"Unique matchups: {num_matchups}")
    console.print(f"Games per matchup: {games_per_matchup}")
    console.print(f"Total games: {total_games}")
    console.print(f"Threads: {threads}")
    
    prompt_files = {
        "clue_giver_prompt": clue_giver_prompt,
        "guesser_prompt": guesser_prompt,
    }
    
    run_id = f"eval_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}"
    
    # Initialize controllog
    try:
        cl.init(project_id="chainlex", log_dir=log_dir)
        cl.event(
            kind="run_start",
            actor={"agent_id": "agent:chainlex"},
            run_id=run_id,
            payload={
                "version": ChainLexGame.VERSION,
                "seed": seed,
                "threads": threads,
                "total_games": total_games,
                "models": eval_models,
            },
            project_id="chainlex",
            source="runtime",
        )
    except Exception:
        pass
    
    results: List[Dict] = []
    failed: List[Dict] = []
    total_cost = 0.0
    lock = threading.Lock()
    
    console.print(f"\n[bold]Starting evaluation...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running games...", total=total_games)
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {}
            for matchup in matchups:
                future = executor.submit(
                    _run_single_game,
                    matchup["model_a"],
                    matchup["model_b"],
                    matchup["seed"],
                    words_file,
                    log_dir,
                    run_id,
                    prompt_files,
                    lock,
                )
                futures[future] = matchup
            
            completed = 0
            for future in as_completed(futures):
                matchup = futures[future]
                game_id, result, error = future.result()
                completed += 1
                
                if result:
                    with lock:
                        results.append(result)
                        game_cost = result.get("cost", 0) + result.get("upstream_cost", 0)
                        total_cost += game_cost
                    
                    winner = result.get("winner_model") or "TIE"
                    score_a = result.get("score_a", 0)
                    score_b = result.get("score_b", 0)
                    
                    msg = f"[green]✅ {completed}/{total_games} | {matchup['model_a']} vs {matchup['model_b']} | {score_a}-{score_b} | Winner: {winner} | ${game_cost:.4f}[/green]"
                    progress.console.print(msg)
                else:
                    with lock:
                        failed.append({"matchup": matchup, "error": error})
                    
                    msg = f"[red]❌ {completed}/{total_games} | {matchup['model_a']} vs {matchup['model_b']} | FAILED: {error}[/red]"
                    progress.console.print(msg)
                
                progress.advance(task)
    
    # Generate summary
    console.print(f"\n[bold]Evaluation Complete![/bold]")
    console.print(f"Total cost: ${total_cost:.2f}")
    
    # Calculate win/loss records
    records: Dict[str, Dict[str, int]] = {m: {"wins": 0, "losses": 0, "ties": 0, "total_score": 0, "games": 0} for m in eval_models}
    
    for r in results:
        model_a = r["model_a"]
        model_b = r["model_b"]
        winner = r["winner"]
        
        records[model_a]["games"] += 1
        records[model_b]["games"] += 1
        records[model_a]["total_score"] += r["score_a"]
        records[model_b]["total_score"] += r["score_b"]
        
        if winner == "model_a":
            records[model_a]["wins"] += 1
            records[model_b]["losses"] += 1
        elif winner == "model_b":
            records[model_b]["wins"] += 1
            records[model_a]["losses"] += 1
        else:
            records[model_a]["ties"] += 1
            records[model_b]["ties"] += 1
    
    # Summary table
    summary_table = Table(title="ChainLex-1 Round-Robin Results")
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("W", justify="right", style="green")
    summary_table.add_column("L", justify="right", style="red")
    summary_table.add_column("T", justify="right", style="yellow")
    summary_table.add_column("Win %", justify="right")
    summary_table.add_column("Avg Score", justify="right")
    
    # Sort by wins
    sorted_models = sorted(eval_models, key=lambda m: (records[m]["wins"], -records[m]["losses"]), reverse=True)
    
    for model in sorted_models:
        r = records[model]
        win_pct = r["wins"] / r["games"] * 100 if r["games"] > 0 else 0
        avg_score = r["total_score"] / r["games"] if r["games"] > 0 else 0
        
        summary_table.add_row(
            model,
            str(r["wins"]),
            str(r["losses"]),
            str(r["ties"]),
            f"{win_pct:.1f}%",
            f"{avg_score:.1f}",
        )
    
    console.print(summary_table)
    
    # Save Bradley-Terry compatible results
    bt_results_path = output / "results.csv"
    with open(bt_results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model_a", "model_b", "winner"])
        
        for r in results:
            writer.writerow([r["model_a"], r["model_b"], r["winner"]])
    
    console.print(f"\n[green]📊 Bradley-Terry results saved to: {bt_results_path}[/green]")
    
    # Save detailed results
    detailed_path = output / "detailed_results.csv"
    with open(detailed_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["game_id", "model_a", "model_b", "score_a", "score_b", "winner", "margin", "cost"])
        
        for r in results:
            writer.writerow([
                r["game_id"],
                r["model_a"],
                r["model_b"],
                r["score_a"],
                r["score_b"],
                r["winner"],
                r["margin"],
                r.get("cost", 0) + r.get("upstream_cost", 0),
            ])
    
    console.print(f"[green]📊 Detailed results saved to: {detailed_path}[/green]")
    
    # Report failures
    if failed:
        console.print(f"\n[yellow]⚠️ {len(failed)} games failed[/yellow]")


@app.command()
def list_canonical():
    """List canonical models for evaluation."""
    canonical_models = _load_canonical_models()
    model_mappings = _load_model_mappings()
    
    if not canonical_models:
        console.print("[yellow]No canonical models defined[/yellow]")
        return
    
    table = Table(title="Canonical Models for ChainLex-1")
    table.add_column("#", style="dim", justify="right")
    table.add_column("CLI Name", style="cyan")
    table.add_column("OpenRouter Model ID", style="magenta")
    
    for i, model_name in enumerate(canonical_models, 1):
        model_id = model_mappings.get(model_name, "UNKNOWN")
        table.add_row(str(i), model_name, model_id)
    
    console.print(table)
    console.print(f"\n✨ Total: {len(canonical_models)} canonical models")
    console.print(f"\n💡 Run evaluation: [bold]uv run based chainlex eval --all[/bold]")


@app.command()
def optimize(
    output: Path = typer.Option(
        Path("chainlex/optimized_prompts"),
        "--output", "-o",
        help="Output directory for optimized pipeline and prompts",
    ),
    model: str = typer.Option(
        "gemini-3-flash",
        "--model", "-m",
        help="Model to use for optimization (CLI name)",
    ),
    num_train: int = typer.Option(
        30,
        "--num-train",
        help="Number of training boards",
    ),
    num_eval: int = typer.Option(
        10,
        "--num-eval",
        help="Number of evaluation boards",
    ),
    threads: int = typer.Option(
        4,
        "--threads", "-t",
        help="Parallel evaluation threads",
    ),
    max_demos: int = typer.Option(
        4,
        "--max-demos",
        help="Maximum few-shot demonstrations to bootstrap",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Test pipeline without full optimization",
    ),
    clue_giver_prompt: Optional[Path] = typer.Option(
        None, "--clue-giver-prompt", help="Custom clue giver prompt file"
    ),
    guesser_prompt: Optional[Path] = typer.Option(
        None, "--guesser-prompt", help="Custom guesser prompt file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Optimize prompts using DSPy GEPA (Genetic Evolution of Prompts Algorithm).
    
    GEPA uses evolutionary optimization with textual feedback to evolve
    better instructions. It reflects on failures and proposes improvements.
    
    Examples:
        # Dry run to test setup
        uv run based chainlex optimize --dry-run
        
        # Full optimization with GEPA
        uv run based chainlex optimize --num-train 50 --num-eval 15
        
        # Use specific model
        uv run based chainlex optimize --model claude-3.5-sonnet
    """
    from chainlex.optimization.optimize import run_optimization, export_optimized_prompts
    
    # Validate API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    
    # Validate model
    model_mappings = _load_model_mappings()
    if model not in model_mappings:
        console.print(f"[red]Error: Invalid model '{model}'[/red]")
        console.print(f"Available models: {', '.join(sorted(model_mappings.keys()))}")
        raise typer.Exit(1)
    
    openrouter_model = model_mappings[model]
    
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    console.print("[bold blue]🧪 ChainLex-1 DSPy Optimization[/bold blue]")
    console.print(f"Model: {model} ({openrouter_model})")
    console.print(f"Training examples: {num_train}")
    console.print(f"Evaluation examples: {num_eval}")
    console.print(f"Max demos: {max_demos}")
    console.print(f"Output: {output}")
    
    if dry_run:
        console.print("[yellow]DRY RUN - testing pipeline only[/yellow]")
    
    console.print()
    
    try:
        results = run_optimization(
            output_dir=output,
            model=openrouter_model,
            num_train_examples=num_train,
            num_eval_examples=num_eval,
            num_threads=threads,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
            seed=seed,
            clue_giver_prompt=str(clue_giver_prompt) if clue_giver_prompt else None,
            guesser_prompt=str(guesser_prompt) if guesser_prompt else None,
            dry_run=dry_run,
        )
        
        if dry_run:
            console.print("[bold green]✅ Dry run completed![/bold green]")
            console.print(f"Test clue: {results['test_clue']} ({results.get('test_number', '?')})")
            console.print(f"Test guesses: {results.get('test_guesses', [])}")
            console.print(f"Test score: {results['test_score']}")
        else:
            console.print("\n[bold green]✅ Optimization complete![/bold green]")
            console.print(f"Baseline score: {results['baseline_score_raw']:.1f} (normalized: {results['baseline_score_normalized']:.3f})")
            console.print(f"Optimized score: {results['optimized_score_raw']:.1f} (normalized: {results['optimized_score_normalized']:.3f})")
            console.print(f"Improvement: {results['improvement_raw']:+.1f} ({results['improvement_pct_raw']:+.1f}%)")
            console.print(f"\nPipeline saved to: {output / 'optimized_pipeline.json'}")
            
            # Export optimized prompts back to markdown
            console.print("\nExporting optimized prompts to markdown...")
            export_optimized_prompts(
                pipeline_path=output / "optimized_pipeline.json",
                output_dir=output,
            )
            console.print(f"[green]Optimized prompts saved to: {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during optimization: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def deploy_prompts(
    optimized_dir: Path = typer.Option(
        Path("chainlex/optimized_prompts"),
        "--from", "-f",
        help="Directory containing optimized prompts",
    ),
    no_backup: bool = typer.Option(
        False,
        "--no-backup",
        help="Don't backup original prompts before overwriting",
    ),
):
    """Deploy optimized prompts to the main prompts folder.
    
    Copies optimized prompts from chainlex/optimized_prompts/ to chainlex/prompts/,
    backing up the originals first.
    
    Examples:
        # Deploy with backup (default)
        uv run based chainlex deploy-prompts
        
        # Deploy without backup
        uv run based chainlex deploy-prompts --no-backup
    """
    from chainlex.optimization.optimize import deploy_optimized_prompts
    
    console.print("[bold blue]🚀 Deploying optimized prompts...[/bold blue]")
    console.print(f"Source: {optimized_dir}")
    console.print(f"Backup: {'No' if no_backup else 'Yes'}")
    console.print()
    
    try:
        deployed = deploy_optimized_prompts(
            optimized_dir=optimized_dir,
            backup=not no_backup,
        )
        
        if deployed:
            console.print("[bold green]✅ Prompts deployed successfully![/bold green]")
            for name, path in deployed.items():
                console.print(f"  • {name} -> {path}")
            
            if not no_backup:
                console.print("\n[dim]Original prompts backed up with .backup extension[/dim]")
                console.print("[dim]Use 'uv run based chainlex rollback-prompts' to restore[/dim]")
        else:
            console.print("[yellow]No prompts were deployed[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error deploying prompts: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def rollback_prompts():
    """Rollback prompts to their original versions.
    
    Restores prompts from .backup files created during deployment.
    
    Examples:
        uv run based chainlex rollback-prompts
    """
    from chainlex.optimization.optimize import rollback_prompts as do_rollback
    
    console.print("[bold blue]⏪ Rolling back prompts...[/bold blue]")
    
    try:
        restored = do_rollback()
        
        if restored:
            console.print("[bold green]✅ Prompts restored successfully![/bold green]")
            for name, path in restored.items():
                console.print(f"  • {name} restored")
        else:
            console.print("[yellow]No backup files found to restore[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error rolling back prompts: {e}[/red]")
        raise typer.Exit(1)


def _run_cost_estimation_game(
    model: str,
    seed: int,
    words_file: str,
    log_dir: Path,
    run_id: str,
    prompt_files: Dict[str, str],
    lock: threading.Lock,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Run a single cost estimation game: model vs gemini-3-flash."""
    try:
        player_a = AIPlayer(model)
        player_b = AIPlayer("gemini-3-flash")
        
        game = ChainLexGame(
            words_file=words_file,
            player_a=player_a,
            player_b=player_b,
            quiet=True,
            seed=seed,
            **prompt_files,
        )
        
        game.init_controllog(log_dir, run_id)
        result = game.play()
        
        return model, result, None
        
    except Exception as e:
        return model, None, str(e)


@app.command("cost-estimate")
def cost_estimate(
    seed: int = typer.Option(42, help="Random seed for board generation"),
    threads: int = typer.Option(8, "--threads", "-t", help="Number of parallel threads"),
    output: Path = typer.Option(
        Path("logs/chainlex/cost_estimate"),
        "--output", "-o",
        help="Output directory for results"
    ),
    words_file: str = typer.Option("inputs/names.yaml", help="Path to words YAML file"),
    clue_giver_prompt: str = typer.Option("chainlex/prompts/clue_giver.md"),
    guesser_prompt: str = typer.Option("chainlex/prompts/guesser.md"),
    games_per_matchup: int = typer.Option(5, help="Games per matchup for projection"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Estimate tournament cost by running each canonical model vs gemini-3-flash.
    
    Runs one game per canonical model against gemini-3-flash (same board for all).
    Then projects total tournament cost based on measured per-model costs.
    """
    canonical_models = _load_canonical_models()
    
    if not canonical_models:
        console.print("[red]Error: No canonical models found[/red]")
        raise typer.Exit(1)
    
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
        raise typer.Exit(1)
    
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir, verbose)
    
    run_id = f"cost_estimate_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}"
    
    # Initialize controllog
    try:
        cl.init(project_id="chainlex", log_dir=log_dir)
        cl.event(
            kind="run_start",
            actor={"agent_id": "agent:chainlex"},
            run_id=run_id,
            payload={
                "version": ChainLexGame.VERSION,
                "seed": seed,
                "threads": threads,
                "total_games": len(canonical_models),
                "estimation_mode": True,
            },
            project_id="chainlex",
            source="runtime",
        )
    except Exception:
        pass
    
    console.print(f"[bold blue]💰 ChainLex-1 Tournament Cost Estimation[/bold blue]")
    console.print(f"Models to test: {len(canonical_models)} canonical models")
    console.print(f"Each model plays 1 game vs gemini-3-flash (same board)")
    console.print(f"Seed: {seed}")
    console.print(f"Threads: {threads}")
    
    prompt_files = {
        "clue_giver_prompt": clue_giver_prompt,
        "guesser_prompt": guesser_prompt,
    }
    
    results: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = {}
    total_cost = 0.0
    lock = threading.Lock()
    
    console.print(f"\n[bold]Running cost estimation games...[/bold]\n")
    
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
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
        
        for future in as_completed(futures):
            model = futures[future]
            model_name, result, error = future.result()
            
            completed_count += 1
            
            if error:
                failed[model_name] = error
                console.print(f"[red]❌ [{completed_count}/{len(canonical_models)}] {model_name}: {error}[/red]")
            else:
                results[model_name] = result
                game_cost = result.get("cost", 0) + result.get("upstream_cost", 0)
                total_cost += game_cost
                
                score_a = result.get("score_a", 0)
                score_b = result.get("score_b", 0)
                
                console.print(
                    f"✓ [{completed_count}/{len(canonical_models)}] [cyan]{model_name}[/cyan] vs gemini-3-flash | "
                    f"Scores: {score_a} vs {score_b} | "
                    f"Cost: ${game_cost:.4f}"
                )
    
    # Summary
    console.print(f"\n[bold]📊 Cost Estimation Results[/bold]\n")
    
    if failed:
        console.print(f"[yellow]⚠️ {len(failed)} models failed[/yellow]")
    
    console.print(f"Successful games: {len(results)}")
    console.print(f"Total estimation cost: ${total_cost:.4f}")
    
    if results:
        avg_cost = total_cost / len(results)
        console.print(f"Average cost per game: ${avg_cost:.4f}")
    
    # Project tournament cost
    console.print(f"\n[bold blue]📈 Tournament Cost Projection[/bold blue]\n")
    
    num_models = len(canonical_models)
    num_matchups = num_models * (num_models - 1) // 2
    total_games = num_matchups * games_per_matchup
    
    console.print(f"Models: {num_models}")
    console.print(f"Unique matchups: {num_matchups}")
    console.print(f"Games per matchup: {games_per_matchup}")
    console.print(f"Total games: {total_games}")
    
    # Build per-model cost
    model_costs = {model: r.get("cost", 0) + r.get("upstream_cost", 0) for model, r in results.items()}
    
    # Estimate gemini-3-flash cost (it was opponent in all games)
    if model_costs:
        avg_game_cost = sum(model_costs.values()) / len(model_costs)
        model_costs["gemini-3-flash"] = avg_game_cost * 0.3
    
    # Project costs
    projected_total = 0.0
    for i, model_a in enumerate(canonical_models):
        for model_b in canonical_models[i + 1:]:
            cost_a = model_costs.get(model_a, avg_cost if results else 0.01)
            cost_b = model_costs.get(model_b, avg_cost if results else 0.01)
            game_cost = (cost_a + cost_b) * 1.2  # 1.2x scaling factor
            projected_total += game_cost * games_per_matchup
    
    console.print(f"\n[bold green]💵 Projected Total Cost: ${projected_total:.2f}[/bold green]")
    
    # Show top 10 most expensive models
    if model_costs:
        table = Table(title="Cost per Model (Top 10)")
        table.add_column("Model", style="cyan")
        table.add_column("Cost/Game", style="yellow", justify="right")
        
        sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
        for model, cost in sorted_models[:10]:
            table.add_row(model, f"${cost:.4f}")
        
        console.print(table)
    
    # Save results
    results_path = output / "cost_estimate.json"
    import json
    with open(results_path, "w") as f:
        json.dump({
            "seed": seed,
            "total_estimation_cost": total_cost,
            "projected_tournament_cost": projected_total,
            "games_per_matchup": games_per_matchup,
            "total_games": total_games,
            "model_costs": model_costs,
            "failed_models": list(failed.keys()),
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {results_path}[/green]")
