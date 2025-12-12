"""Command-line interface for BASED Eval - Multi-game AI evaluation framework.

This is the unified CLI entry point for all BASED eval games:
- `based codenames` - Run Codenames games
- `based connections` - Run Connections puzzles  
- `based analytics` - Analytics and reporting tools
"""

import typer
from rich.console import Console

from codenames.cli_codenames import app as codenames_app
from codenames.cli_connections import app as connections_app
from codenames.cli_analytics import app as analytics_app

# Main application
app = typer.Typer(
    help="BASED Eval - Benchmark for Association, Sorting, and Entity Deduction",
    no_args_is_help=True,
)
console = Console()

# Register subcommands
app.add_typer(codenames_app, name="codenames", help="Run Codenames games for AI evaluation")
app.add_typer(connections_app, name="connections", help="Run Connections puzzles for AI evaluation")
app.add_typer(analytics_app, name="analytics", help="Analytics and reporting tools")


@app.callback()
def main():
    """BASED Eval - Multi-game AI evaluation framework.
    
    Run AI models on various games to evaluate their reasoning and language abilities.
    
    Examples:
    
        # Run a Codenames game
        uv run based codenames run --red gemini-flash --blue claude-haiku
        
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
    
    console.print(f"[bold]BASED Eval[/bold]")
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
