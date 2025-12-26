"""Command-line interface for BASED Eval - Multi-game AI evaluation framework.

This is the unified CLI entry point for all BASED eval games:
- `based codenames` - Run Codenames games
- `based chainlex` - Run ChainLex-1 games (single-player word association)
- `based connections` - Run Connections puzzles  
- `based analytics` - Analytics and reporting tools
"""

import sys
from pathlib import Path

import typer
from rich.console import Console

# Add connections/src to path for connections_eval imports
connections_path = Path(__file__).parent / "connections" / "src"
if str(connections_path) not in sys.path:
    sys.path.insert(0, str(connections_path))

# Import game-specific CLIs
from codenames.cli_codenames import app as codenames_app
from chainlex.cli_chainlex import app as chainlex_app
from connections_eval.cli import app as connections_app
from shared.cli_analytics import app as analytics_app

# Main application
app = typer.Typer(
    help="BASED Eval - Benchmark for Association, Sorting, and Entity Deduction",
    no_args_is_help=True,
)
console = Console()

# Register subcommands
app.add_typer(codenames_app, name="codenames", help="Run Codenames games for AI evaluation")
app.add_typer(chainlex_app, name="chainlex", help="Run ChainLex-1 games (single-player word association)")
app.add_typer(connections_app, name="connections", help="Run Connections puzzles for AI evaluation")
app.add_typer(analytics_app, name="analytics", help="Analytics and reporting tools")


@app.callback()
def main():
    """BASED Eval - Multi-game AI evaluation framework.
    
    Run AI models on various games to evaluate their reasoning and language abilities.
    
    Examples:
    
        # Run a Codenames game
        uv run based codenames run --red gemini-flash --blue claude-haiku
        
        # Run a ChainLex-1 game (single-player, cost-efficient)
        uv run based chainlex run --model gemini-flash
        
        # Run Connections puzzles
        uv run based connections run --model gemini-flash --puzzles 10
        
        # Check analytics
        uv run based analytics trial-balance
        uv run based analytics cost-report
        uv run based analytics leaderboard
    """
    pass


@app.command()
def version():
    """Show version information."""
    from codenames import __version__ as codenames_version
    from shared import __version__ as shared_version
    
    console.print("[bold]BASED Eval[/bold]")
    console.print(f"  codenames: {codenames_version}")
    console.print(f"  shared: {shared_version}")


# Legacy commands for backward compatibility
# These will show a deprecation warning and redirect to the new commands

@app.command(hidden=True)
def run():
    """[DEPRECATED] Use 'based codenames run' instead."""
    console.print("[yellow]⚠️  'based run' is deprecated. Use 'based codenames run' instead.[/yellow]")
    console.print("[dim]Example: uv run based codenames run --red gemini-flash --blue claude-haiku[/dim]")
    raise typer.Exit(1)


@app.command(name="list-models", hidden=True)
def list_models_legacy():
    """[DEPRECATED] Use 'based codenames list-models' or 'based connections list-models' instead."""
    console.print("[yellow]⚠️  'based list-models' is deprecated.[/yellow]")
    console.print("[dim]Use: uv run based codenames list-models[/dim]")
    console.print("[dim]  or: uv run based connections list-models[/dim]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
