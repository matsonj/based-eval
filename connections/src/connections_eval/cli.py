"""CLI interface for connections_eval."""

import os
import sys
import yaml
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import ConnectionsGame, PuzzleDifficultyResult
# Use shared infrastructure from BASED eval framework
from shared.utils.motherduck import (
    upload_controllog_to_motherduck,
    validate_upload,
    run_trial_balance,
    cleanup_local_files,
)

app = typer.Typer(help="Evaluate AI models on New York Times Connections puzzles")
console = Console()


def _resolve_inputs_path(inputs_path: Path, is_default: bool = False) -> Path:
    """Resolve inputs path, checking multiple locations for required files.
    
    Args:
        inputs_path: The path to resolve
        is_default: If True, try alternative locations. If False, use path as-is.
    """
    required_file = "connections_puzzles.yml"
    
    # If the path has the required file, use it
    if (inputs_path / required_file).exists():
        return inputs_path
    
    # Only try alternative locations if using default path
    if not is_default:
        return inputs_path
    
    # Try connections/inputs (when running from monorepo root)
    monorepo_path = Path("connections/inputs")
    if (monorepo_path / required_file).exists():
        return monorepo_path
    
    # Try relative to this file's location
    cli_dir = Path(__file__).parent.parent.parent  # connections/src/connections_eval -> connections
    relative_path = cli_dir / "inputs"
    if (relative_path / required_file).exists():
        return relative_path
    
    # Return original path (will error later with helpful message)
    return inputs_path

def main():
    """Entry point for CLI."""
    app()

@app.command()
def run(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model to evaluate (grok3, grok4, o3, o4-mini, gpt4, gpt4-turbo, gemini, sonnet, opus)"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Run in interactive mode (human player)"
    ),
    puzzles: Optional[int] = typer.Option(
        None,
        "--puzzles",
        help="Maximum number of puzzles to run (default: all)"
    ),
    puzzle_ids: Optional[str] = typer.Option(
        None,
        "--puzzle-ids",
        help="Comma-separated list of specific puzzle IDs to run (e.g., '813,814,821')"
    ),
    canonical: bool = typer.Option(
        False,
        "--canonical",
        help="Use the canonical puzzle set from puzzle_difficulty.yml"
    ),
    threads: int = typer.Option(
        8,
        "--threads", "-t",
        help="Number of parallel threads (default: 8, use 1 for sequential)"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility"
    ),
    inputs_path: Optional[Path] = typer.Option(
        None,
        "--inputs-path",
        help="Path to inputs directory (default: auto-detect)"
    ),
    log_path: Path = typer.Option(
        Path("logs"),
        "--log-path", 
        help="Path to logs directory"
    ),
    prompt_file: str = typer.Option(
        "prompt_template.xml",
        "--prompt-file",
        help="Prompt template file name"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Print logs to terminal for debugging"
    ),
    keep_local_files: bool = typer.Option(
        False,
        "--keep-local-files",
        help="Keep local controllog files after uploading to MotherDuck (default: False, files are deleted)"
    )
):
    """Run connections evaluation."""
    
    # Validate arguments
    if not interactive and not model:
        console.print("❌ Either --model or --interactive must be specified", style="red")
        raise typer.Exit(1)
    
    if interactive and model:
        console.print("❌ Cannot specify both --model and --interactive", style="red") 
        raise typer.Exit(1)
    
    if interactive and threads > 1:
        console.print("⚠️  Interactive mode requires sequential execution, ignoring --threads", style="yellow")
        threads = 1
    
    # Validate puzzle selection options (mutually exclusive)
    options_specified = sum([puzzles is not None, puzzle_ids is not None, canonical])
    if options_specified > 1:
        console.print("❌ Cannot combine --puzzles, --puzzle-ids, and --canonical. Choose one.", style="red")
        raise typer.Exit(1)
    
    # Resolve inputs path (handles running from monorepo root vs connections dir)
    is_default_path = inputs_path is None
    if is_default_path:
        inputs_path = Path("inputs")
    inputs_path = _resolve_inputs_path(inputs_path, is_default=is_default_path)
    
    # Validate inputs path
    if not inputs_path.exists():
        console.print(f"❌ Inputs path does not exist: {inputs_path}", style="red")
        if is_default_path:
            console.print("[dim]Try running from the connections directory or specifying --inputs-path[/dim]")
        raise typer.Exit(1)
    
    # Validate model (create temporary instance to load model config)
    if model:
        try:
            temp_game = ConnectionsGame(inputs_path, log_path)
        except FileNotFoundError as e:
            console.print(f"❌ Error loading model config: {e}", style="red")
            raise typer.Exit(1)
        
        if model not in temp_game.MODEL_CONFIG:
            console.print(f"❌ Unknown model: {model}", style="red")
            console.print("Available models:", style="yellow")
            for model_name in temp_game.MODEL_CONFIG.keys():
                console.print(f"  - {model_name}")
            raise typer.Exit(2)
    
    # Check OpenRouter API key for non-interactive mode
    if not interactive:
        if not os.getenv("OPENROUTER_API_KEY"):
            console.print("❌ OPENROUTER_API_KEY environment variable not set. Try `source .env` and run again.", style="red")
            raise typer.Exit(1)
    
    puzzles_file = inputs_path / "connections_puzzles.yml"
    template_file = inputs_path / prompt_file
    
    if not puzzles_file.exists():
        console.print(f"❌ Puzzles file not found: {puzzles_file}", style="red")
        raise typer.Exit(1)
    
    if not template_file.exists():
        console.print(f"❌ Prompt template not found: {template_file}", style="red")
        raise typer.Exit(1)
    
    # Get model name for interactive mode
    if interactive:
        model_name = typer.prompt("Enter a label for this run (for logging)")
    else:
        model_name = model
    
    # Parse puzzle IDs if specified
    parsed_puzzle_ids: Optional[List[int]] = None
    
    if puzzle_ids:
        try:
            parsed_puzzle_ids = [int(pid.strip()) for pid in puzzle_ids.split(',')]
            console.print(f"📋 Running specific puzzles: {parsed_puzzle_ids}", style="dim")
        except ValueError:
            console.print(f"❌ Invalid puzzle IDs format. Use comma-separated integers: '813,814,821'", style="red")
            raise typer.Exit(1)
    
    if canonical:
        # Load canonical puzzle set from connections_puzzles.yml (puzzles with canonical: true)
        try:
            temp_game = ConnectionsGame(inputs_path, log_path)
            parsed_puzzle_ids = temp_game.get_canonical_puzzle_ids()
            
            if not parsed_puzzle_ids:
                console.print(f"❌ No canonical puzzles defined in connections_puzzles.yml", style="red")
                console.print("[dim]Mark puzzles with 'canonical: true' in the puzzle file[/dim]")
                raise typer.Exit(1)
            
            console.print(f"📋 Using canonical puzzle set ({len(parsed_puzzle_ids)} puzzles)", style="dim")
        except Exception as e:
            console.print(f"❌ Error loading canonical puzzle set: {e}", style="red")
            raise typer.Exit(1)
    
    # Initialize game
    try:
        game = ConnectionsGame(inputs_path, log_path, seed, verbose=verbose)
        
        # Determine puzzle count for display
        if parsed_puzzle_ids:
            puzzle_count_str = f"{len(parsed_puzzle_ids)} specific puzzles"
        elif puzzles:
            puzzle_count_str = str(puzzles)
        else:
            puzzle_count_str = "all"
        
        # Show run info
        console.print(f"🎮 Starting Connections evaluation", style="bold blue")
        console.print(f"Mode: {'Interactive' if interactive else f'AI Model ({model})'}")
        console.print(f"Puzzles: {puzzle_count_str}")
        console.print(f"Threads: {threads}")
        console.print(f"Seed: {game.seed}")
        console.print(f"Log path: {log_path}")
        console.print()
        
        # Run evaluation
        summary = game.run_evaluation(
            model_name, 
            puzzles, 
            interactive,
            threads=threads,
            puzzle_ids=parsed_puzzle_ids
        )
        
        # Display results
        _display_summary(summary, interactive)
        
        # Upload to MotherDuck if configured
        motherduck_db = os.getenv("MOTHERDUCK_DB")
        if motherduck_db:
            console.print()
            console.print("📤 Uploading controllog to MotherDuck...", style="bold blue")
            
            # Use CTRL_LOG_DIR if set, otherwise use log_path
            ctrl_log_dir = Path(os.getenv("CTRL_LOG_DIR", str(log_path)))
            
            # Upload
            upload_success = upload_controllog_to_motherduck(ctrl_log_dir, motherduck_db)
            if upload_success:
                console.print("✅ Upload successful", style="green")
                
                # Validate upload
                console.print("🔍 Validating upload...", style="dim")
                validation_success = validate_upload(summary["run_id"], motherduck_db)
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
                    cleanup_local_files(ctrl_log_dir, summary["run_id"], keep_local_files)
                    console.print("✅ Local files cleaned up", style="green")
                else:
                    console.print("📁 Keeping local files (--keep-local-files flag set)", style="dim")
            else:
                console.print("❌ Upload failed", style="red")
                console.print("⚠️  Local files retained due to upload failure", style="yellow")
        
    except KeyboardInterrupt:
        console.print("\n❌ Evaluation interrupted", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


def _display_summary(summary: dict, interactive: bool):
    """Display evaluation summary."""
    console.print("📊 Evaluation Results", style="bold green")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Model/Run", summary["model"])
    table.add_row("Version", summary.get("version", "unknown"))
    table.add_row("Puzzles Attempted", str(summary["puzzles_attempted"]))
    table.add_row("Puzzles Solved", str(summary["puzzles_solved"]))
    
    if summary["puzzles_attempted"] > 0:
        solve_rate = summary["puzzles_solved"] / summary["puzzles_attempted"] * 100
        table.add_row("Solve Rate", f"{solve_rate:.1f}%")
    
    table.add_row("Total Guesses", str(summary["total_guesses"]))
    table.add_row("Correct Guesses", str(summary["correct_guesses"]))
    table.add_row("Incorrect Guesses", str(summary["incorrect_guesses"]))
    table.add_row("Invalid Responses", str(summary["invalid_responses"]))
    
    if summary["total_guesses"] > 0:
        accuracy = summary["correct_guesses"] / summary["total_guesses"] * 100
        table.add_row("Guess Accuracy", f"{accuracy:.1f}%")
    
    table.add_row("Average Time", f"{summary['avg_time_sec']:.1f}s")
    
    if not interactive:
        table.add_row("Total Tokens", str(summary["total_tokens"]))
        if summary.get("total_prompt_tokens", 0) > 0:
            table.add_row("Prompt Tokens", str(summary["total_prompt_tokens"]))
        if summary.get("total_completion_tokens", 0) > 0:
            table.add_row("Completion Tokens", str(summary["total_completion_tokens"]))
        table.add_row("Token Method", summary["token_count_method"])
        
        # Add cost information
        if summary.get("total_cost", 0) > 0:
            table.add_row("OpenRouter Cost", f"${summary['total_cost']:.6f}")
        if summary.get("total_upstream_cost", 0) > 0:
            table.add_row("OpenAI Cost", f"${summary['total_upstream_cost']:.6f}")
    
    table.add_row("Seed", str(summary["seed"]))
    table.add_row("Threads", str(summary.get("threads", 1)))
    
    console.print(table)
    
    # Show puzzle IDs if tracked
    puzzle_ids = summary.get("puzzle_ids", [])
    if puzzle_ids:
        console.print(f"\n📋 Puzzles run: {', '.join(str(p) for p in sorted(puzzle_ids))}", style="dim")
    
    console.print()
    
    # Show log file location
    console.print(f"📝 Detailed logs saved to: {summary['run_id']}", style="dim")


@app.command()
def rank(
    puzzle_id: Optional[int] = typer.Option(
        None,
        "--puzzle-id", "-p",
        help="Specific puzzle ID to rank (omit to rank all puzzles)"
    ),
    runs: int = typer.Option(
        10,
        "--runs", "-r",
        help="Number of times to run each puzzle (default: 10)"
    ),
    model: str = typer.Option(
        "gemini-3-flash",
        "--model", "-m",
        help="Model to use for ranking (default: gemini-3-flash)"
    ),
    threads: int = typer.Option(
        4,
        "--threads", "-t",
        help="Number of parallel threads for ranking all puzzles (default: 4)"
    ),
    inputs_path: Optional[Path] = typer.Option(
        None,
        "--inputs-path",
        help="Path to inputs directory (default: auto-detect)"
    ),
    log_path: Path = typer.Option(
        Path("logs"),
        "--log-path",
        help="Path to logs directory"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for difficulty rankings (default: inputs/puzzle_difficulty.yml)"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Base random seed for reproducibility (default: 42)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Print detailed logs"
    )
):
    """
    Rank puzzle difficulty by running puzzles multiple times.
    
    This command evaluates puzzle difficulty by running each puzzle multiple times
    on a consistent model (default: gemini-3-flash) and measuring:
    - Win rate (lower = harder)
    - Completion tokens used (more = harder)
    - Guess counts and mistake rates
    
    Results are saved to a YAML file that can be used to define the canonical puzzle set.
    
    Examples:
        # Rank a single puzzle
        based connections rank --puzzle-id 813 --runs 10
        
        # Rank all puzzles with 4 parallel threads
        based connections rank --threads 4 --runs 5
    """
    # Check OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("❌ OPENROUTER_API_KEY environment variable not set", style="red")
        raise typer.Exit(1)
    
    # Resolve inputs path
    is_default_path = inputs_path is None
    if is_default_path:
        inputs_path = Path("inputs")
    inputs_path = _resolve_inputs_path(inputs_path, is_default=is_default_path)
    
    if not inputs_path.exists():
        console.print(f"❌ Inputs path does not exist: {inputs_path}", style="red")
        raise typer.Exit(1)
    
    # Set default output file
    if output_file is None:
        output_file = inputs_path / "puzzle_difficulty.yml"
    
    # Initialize game
    try:
        game = ConnectionsGame(inputs_path, log_path, seed, verbose=verbose)
    except Exception as e:
        console.print(f"❌ Error initializing game: {e}", style="red")
        raise typer.Exit(1)
    
    # Validate model
    if model not in game.MODEL_CONFIG:
        console.print(f"❌ Unknown model: {model}", style="red")
        console.print("Available models:", style="yellow")
        for model_name in sorted(game.MODEL_CONFIG.keys()):
            console.print(f"  - {model_name}")
        raise typer.Exit(2)
    
    if puzzle_id is not None:
        # Rank single puzzle
        console.print(f"🎯 Ranking puzzle {puzzle_id} ({runs} runs with {model})", style="bold blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Running puzzle {puzzle_id}...", total=None)
            
            try:
                result = game.rank_puzzle(puzzle_id, runs, model)
                progress.update(task, completed=True)
            except ValueError as e:
                console.print(f"❌ {e}", style="red")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"❌ Error ranking puzzle: {e}", style="red")
                raise typer.Exit(1)
        
        results = [result]
    else:
        # Rank all puzzles
        puzzle_count = len(game.puzzles)
        console.print(f"🎯 Ranking all {puzzle_count} puzzles ({runs} runs each, {threads} threads, {model})", style="bold blue")
        console.print(f"   This will make approximately {puzzle_count * runs * 4} API calls", style="dim")
        
        try:
            results = game.rank_all_puzzles(runs, model, threads)
        except Exception as e:
            console.print(f"❌ Error ranking puzzles: {e}", style="red")
            raise typer.Exit(1)
    
    # Display results
    _display_ranking_results(results)
    
    # Save results
    _save_ranking_results(results, output_file)
    console.print(f"\n📁 Results saved to: {output_file}", style="green")


def _display_ranking_results(results: List[PuzzleDifficultyResult]):
    """Display ranking results in a table."""
    console.print("\n📊 Puzzle Difficulty Rankings (hardest first)", style="bold green")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("ID", style="white", justify="right")
    table.add_column("Date", style="dim")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Guesses", justify="right")
    table.add_column("Avg Mistakes", justify="right")
    table.add_column("Difficulty", justify="right")
    table.add_column("Tier", style="bold")
    
    for i, result in enumerate(results, 1):
        # Determine tier based on win rate
        if result.win_rate < 0.5:
            tier = "[red]HARD[/red]"
        elif result.win_rate < 0.8:
            tier = "[yellow]MEDIUM[/yellow]"
        else:
            tier = "[green]EASY[/green]"
        
        table.add_row(
            str(i),
            str(result.puzzle_id),
            str(result.puzzle_date),
            f"{result.win_rate * 100:.0f}%",
            f"{result.avg_completion_tokens:.0f}",
            f"{result.avg_guesses:.1f}",
            f"{result.avg_mistakes:.1f}",
            f"{result.difficulty_score:.1f}",
            tier
        )
    
    console.print(table)
    
    # Summary stats
    hard_count = sum(1 for r in results if r.win_rate < 0.5)
    medium_count = sum(1 for r in results if 0.5 <= r.win_rate < 0.8)
    easy_count = sum(1 for r in results if r.win_rate >= 0.8)
    
    console.print(f"\nTier distribution: [red]{hard_count} hard[/red], [yellow]{medium_count} medium[/yellow], [green]{easy_count} easy[/green]")


def _save_ranking_results(results: List[PuzzleDifficultyResult], output_file: Path):
    """Save ranking results to YAML file."""
    # Load existing results if file exists
    existing_results = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            data = yaml.safe_load(f) or {}
            existing_results = {r['puzzle_id']: r for r in data.get('rankings', [])}
    
    # Update with new results
    for result in results:
        existing_results[result.puzzle_id] = asdict(result)
    
    # Sort by difficulty score (hardest first)
    sorted_results = sorted(existing_results.values(), key=lambda r: r['difficulty_score'])
    
    # Create tier lists for canonical set selection
    hard_tier = [r['puzzle_id'] for r in sorted_results if r['win_rate'] < 0.5]
    medium_tier = [r['puzzle_id'] for r in sorted_results if 0.5 <= r['win_rate'] < 0.8]
    easy_tier = [r['puzzle_id'] for r in sorted_results if r['win_rate'] >= 0.8]
    
    output_data = {
        'metadata': {
            'description': 'Puzzle difficulty rankings based on empirical testing',
            'ranking_model': 'gemini-3-flash',
            'last_updated': results[0].ranked_at if results else None,
        },
        'tiers': {
            'hard': hard_tier,
            'medium': medium_tier,
            'easy': easy_tier,
        },
        'suggested_canonical_set': {
            'description': 'Suggested 20-puzzle canonical set (5 hard, 10 medium, 5 easy)',
            'hard': hard_tier[:5],
            'medium': medium_tier[:10],
            'easy': easy_tier[:5],
        },
        'rankings': sorted_results,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)


@app.command()
def list_puzzles(
    inputs_path: Optional[Path] = typer.Option(
        None,
        "--inputs-path",
        help="Path to inputs directory (default: auto-detect)"
    ),
    show_difficulty: bool = typer.Option(
        False,
        "--difficulty", "-d",
        help="Show difficulty rankings if available"
    )
):
    """List all available puzzles."""
    # Resolve inputs path
    is_default_path = inputs_path is None
    if is_default_path:
        inputs_path = Path("inputs")
    inputs_path = _resolve_inputs_path(inputs_path, is_default=is_default_path)
    
    if not inputs_path.exists():
        console.print(f"❌ Inputs path does not exist: {inputs_path}", style="red")
        raise typer.Exit(1)
    
    # Load puzzles
    puzzles_file = inputs_path / "connections_puzzles.yml"
    if not puzzles_file.exists():
        console.print(f"❌ Puzzles file not found: {puzzles_file}", style="red")
        raise typer.Exit(1)
    
    with open(puzzles_file, 'r') as f:
        data = yaml.safe_load(f)
    
    puzzles = data.get('puzzles', [])
    
    # Load difficulty rankings if requested
    difficulty_data = {}
    if show_difficulty:
        difficulty_file = inputs_path / "puzzle_difficulty.yml"
        if difficulty_file.exists():
            with open(difficulty_file, 'r') as f:
                diff_data = yaml.safe_load(f)
            difficulty_data = {r['puzzle_id']: r for r in diff_data.get('rankings', [])}
    
    console.print(f"📋 Available Puzzles ({len(puzzles)} total)", style="bold blue")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Date", style="white")
    table.add_column("Base Difficulty", justify="right")
    
    if show_difficulty and difficulty_data:
        table.add_column("Win Rate", justify="right")
        table.add_column("Tier", style="bold")
    
    for puzzle in sorted(puzzles, key=lambda p: p['id']):
        row = [
            str(puzzle['id']),
            puzzle['date'],
            f"{puzzle['difficulty']:.1f}",
        ]
        
        if show_difficulty and difficulty_data:
            if puzzle['id'] in difficulty_data:
                diff = difficulty_data[puzzle['id']]
                win_rate = diff['win_rate']
                if win_rate < 0.5:
                    tier = "[red]HARD[/red]"
                elif win_rate < 0.8:
                    tier = "[yellow]MEDIUM[/yellow]"
                else:
                    tier = "[green]EASY[/green]"
                row.extend([f"{win_rate * 100:.0f}%", tier])
            else:
                row.extend(["--", "[dim]unranked[/dim]"])
        
        table.add_row(*row)
    
    console.print(table)
    
    # Show hint about ranking
    if show_difficulty and not difficulty_data:
        console.print("\n[dim]Run 'based connections rank' to generate difficulty rankings[/dim]")


@app.command()
def list_models():
    """List available models."""
    from shared.adapters.openrouter_adapter import _load_model_mappings
    
    console.print("Available models:", style="bold blue")
    
    # Load model mappings from shared infrastructure
    mappings = _load_model_mappings()
    
    # Flatten and display
    all_models = {}
    if "thinking" in mappings:
        all_models.update(mappings["thinking"])
    if "non_thinking" in mappings:
        all_models.update(mappings["non_thinking"])
    
    for model_name in sorted(all_models.keys()):
        console.print(f"  {model_name}")


if __name__ == "__main__":
    main()
