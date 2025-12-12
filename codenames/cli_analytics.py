"""CLI subcommand for shared analytics commands."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Analytics commands for BASED eval framework")
console = Console()


@app.command()
def trial_balance(
    db: Optional[str] = typer.Option(
        None,
        "--db",
        help="MotherDuck database connection string (defaults to MOTHERDUCK_DB env var)"
    ),
):
    """Run trial balance check on controllog data.
    
    Validates that all balanced postings sum to zero (double-entry accounting).
    """
    from shared.utils.motherduck import run_trial_balance
    
    motherduck_db = db or os.getenv("MOTHERDUCK_DB")
    if not motherduck_db:
        console.print("❌ No database specified. Use --db or set MOTHERDUCK_DB env var", style="red")
        raise typer.Exit(1)
    
    console.print(f"⚖️  Running trial balance check on {motherduck_db}...", style="bold blue")
    
    success = run_trial_balance(motherduck_db)
    if success:
        console.print("✅ Trial balance passed - all accounts balanced", style="green")
    else:
        console.print("❌ Trial balance failed - some accounts are unbalanced", style="red")
        raise typer.Exit(1)


@app.command()
def cost_report(
    db: Optional[str] = typer.Option(
        None,
        "--db",
        help="MotherDuck database connection string (defaults to MOTHERDUCK_DB env var)"
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Filter by project (codenames, connections, etc.)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Filter by model name"
    ),
    days: int = typer.Option(
        30,
        "--days",
        help="Number of days to include in report"
    ),
):
    """Show cost breakdown by model and game.
    
    Queries controllog postings to calculate total costs.
    """
    try:
        import duckdb
    except ImportError:
        console.print("❌ duckdb not installed. Install with: pip install duckdb", style="red")
        raise typer.Exit(1)
    
    motherduck_db = db or os.getenv("MOTHERDUCK_DB")
    if not motherduck_db:
        console.print("❌ No database specified. Use --db or set MOTHERDUCK_DB env var", style="red")
        raise typer.Exit(1)
    
    console.print(f"💰 Generating cost report from {motherduck_db}...", style="bold blue")
    
    try:
        con = duckdb.connect(motherduck_db)
        
        # Build query with optional filters
        where_clauses = [f"e.event_time >= NOW() - INTERVAL '{days} days'"]
        if project:
            where_clauses.append(f"e.project_id = '{project}'")
        
        where_sql = " AND ".join(where_clauses)
        
        # Query for cost by model
        query = f"""
        SELECT 
            e.project_id,
            p.dims_json->>'model' as model,
            SUM(CASE WHEN p.account_type = 'resource.money' AND p.delta_numeric > 0 THEN p.delta_numeric ELSE 0 END) as total_cost,
            SUM(CASE WHEN p.account_type = 'resource.tokens' AND p.delta_numeric > 0 THEN p.delta_numeric ELSE 0 END) as total_tokens,
            COUNT(DISTINCT e.event_id) as num_events
        FROM controllog.events e
        JOIN controllog.postings p ON e.event_id = p.event_id
        WHERE {where_sql}
        GROUP BY e.project_id, p.dims_json->>'model'
        ORDER BY total_cost DESC
        """
        
        results = con.execute(query).fetchall()
        
        if not results:
            console.print("No data found for the specified filters", style="yellow")
            con.close()
            return
        
        # Create table
        table = Table(title=f"Cost Report (Last {days} Days)")
        table.add_column("Project", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Cost", style="green", justify="right")
        table.add_column("Tokens", style="blue", justify="right")
        table.add_column("Events", style="dim", justify="right")
        
        total_cost = 0.0
        total_tokens = 0
        for row in results:
            project_id, model_name, cost, tokens, events = row
            total_cost += cost or 0
            total_tokens += int(tokens or 0)
            table.add_row(
                project_id or "unknown",
                model_name or "unknown",
                f"${cost:.6f}" if cost else "$0.00",
                f"{int(tokens or 0):,}",
                str(events)
            )
        
        console.print(table)
        console.print()
        console.print(f"💵 Total Cost: ${total_cost:.6f}", style="bold green")
        console.print(f"📊 Total Tokens: {total_tokens:,}", style="bold blue")
        
        con.close()
        
    except Exception as e:
        console.print(f"❌ Error generating report: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def leaderboard(
    db: Optional[str] = typer.Option(
        None,
        "--db",
        help="MotherDuck database connection string (defaults to MOTHERDUCK_DB env var)"
    ),
    game: str = typer.Option(
        "all",
        "--game",
        help="Game to show leaderboard for (codenames, connections, or all)"
    ),
    days: int = typer.Option(
        30,
        "--days",
        help="Number of days to include in leaderboard"
    ),
):
    """Show model performance leaderboard across games.
    
    Queries controllog events to calculate win rates and scores.
    """
    try:
        import duckdb
    except ImportError:
        console.print("❌ duckdb not installed. Install with: pip install duckdb", style="red")
        raise typer.Exit(1)
    
    motherduck_db = db or os.getenv("MOTHERDUCK_DB")
    if not motherduck_db:
        console.print("❌ No database specified. Use --db or set MOTHERDUCK_DB env var", style="red")
        raise typer.Exit(1)
    
    console.print(f"🏆 Generating leaderboard from {motherduck_db}...", style="bold blue")
    
    try:
        con = duckdb.connect(motherduck_db)
        
        # Build query with optional filters
        where_clauses = [f"event_time >= NOW() - INTERVAL '{days} days'"]
        if game != "all":
            where_clauses.append(f"project_id = '{game}'")
        
        where_sql = " AND ".join(where_clauses)
        
        # Query for state transitions to calculate win/loss rates
        query = f"""
        SELECT 
            project_id,
            payload_json->>'model' as model,
            SUM(CASE WHEN kind = 'state_move' AND payload_json->>'to' = 'DONE' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN kind = 'state_move' AND payload_json->>'to' = 'FAILED' THEN 1 ELSE 0 END) as losses,
            COUNT(DISTINCT run_id) as total_runs
        FROM controllog.events
        WHERE {where_sql}
            AND kind = 'state_move'
        GROUP BY project_id, payload_json->>'model'
        HAVING total_runs > 0
        ORDER BY wins DESC, losses ASC
        """
        
        results = con.execute(query).fetchall()
        
        if not results:
            console.print("No data found for the specified filters", style="yellow")
            console.print("[dim]Note: Leaderboard requires state_move events with DONE/FAILED transitions[/dim]")
            con.close()
            return
        
        # Create table
        table = Table(title=f"Model Leaderboard (Last {days} Days)")
        table.add_column("Rank", style="yellow", justify="right")
        table.add_column("Game", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Wins", style="green", justify="right")
        table.add_column("Losses", style="red", justify="right")
        table.add_column("Win Rate", style="blue", justify="right")
        
        for i, row in enumerate(results, 1):
            project_id, model_name, wins, losses, total_runs = row
            total_games = (wins or 0) + (losses or 0)
            win_rate = (wins or 0) / total_games * 100 if total_games > 0 else 0
            
            table.add_row(
                str(i),
                project_id or "unknown",
                model_name or "unknown",
                str(wins or 0),
                str(losses or 0),
                f"{win_rate:.1f}%"
            )
        
        console.print(table)
        
        con.close()
        
    except Exception as e:
        console.print(f"❌ Error generating leaderboard: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def upload(
    log_path: Path = typer.Option(
        Path("logs"),
        "--log-path",
        help="Path to logs directory containing controllog files"
    ),
    db: Optional[str] = typer.Option(
        None,
        "--db",
        help="MotherDuck database connection string (defaults to MOTHERDUCK_DB env var)"
    ),
    keep_local_files: bool = typer.Option(
        False,
        "--keep-local-files",
        help="Keep local controllog files after uploading"
    ),
):
    """Upload local controllog files to MotherDuck.
    
    Uploads events.jsonl and postings.jsonl files from the controllog directory.
    """
    from shared.utils.motherduck import upload_controllog_to_motherduck, run_trial_balance
    
    motherduck_db = db or os.getenv("MOTHERDUCK_DB")
    if not motherduck_db:
        console.print("❌ No database specified. Use --db or set MOTHERDUCK_DB env var", style="red")
        raise typer.Exit(1)
    
    # Check if controllog directory exists
    controllog_dir = log_path / "controllog"
    if not controllog_dir.exists():
        console.print(f"❌ No controllog directory found at {controllog_dir}", style="red")
        raise typer.Exit(1)
    
    console.print(f"📤 Uploading controllog from {log_path} to {motherduck_db}...", style="bold blue")
    
    success = upload_controllog_to_motherduck(log_path, motherduck_db)
    if success:
        console.print("✅ Upload successful", style="green")
        
        # Run trial balance
        console.print("⚖️  Running trial balance check...", style="dim")
        trial_balance_success = run_trial_balance(motherduck_db)
        if trial_balance_success:
            console.print("✅ Trial balance passed", style="green")
        else:
            console.print("⚠️  Trial balance check failed", style="yellow")
        
        # Cleanup if requested
        if not keep_local_files:
            console.print("🧹 Cleaning up local files...", style="dim")
            # Remove all files in controllog directory
            import shutil
            try:
                shutil.rmtree(controllog_dir)
                console.print("✅ Local files cleaned up", style="green")
            except Exception as e:
                console.print(f"⚠️  Could not clean up files: {e}", style="yellow")
        else:
            console.print("📁 Keeping local files (--keep-local-files flag set)", style="dim")
    else:
        console.print("❌ Upload failed", style="red")
        raise typer.Exit(1)

