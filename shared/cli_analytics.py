"""CLI subcommand for shared analytics commands."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Analytics commands for BASED eval framework")
console = Console()


def plot_leaderboard(
    results: dict,
    output_path: Path,
    top_n: int = 20,
    title: str = "Bradley-Terry Ratings",
    item_name: str = "Model",
    rating_name: str = "BT Rating",
):
    """Generate a leaderboard chart similar to arena-rank examples.
    
    Creates a horizontal bar chart with confidence intervals.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError:
        console.print("[yellow]Warning: matplotlib not installed. Chart not generated.[/yellow]")
        console.print("[dim]Install with: uv add matplotlib[/dim]")
        return False
    
    # Create DataFrame and sort
    df = pd.DataFrame(results).sort_values("ratings", ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get data
    competitors = df["competitors"].values
    ratings = df["ratings"].values
    lower = df["rating_lower"].values
    upper = df["rating_upper"].values
    
    # Calculate error bars
    errors_lower = ratings - lower
    errors_upper = upper - ratings
    errors = np.array([errors_lower, errors_upper])
    
    # Y positions (reversed so highest rating is at top)
    y_pos = np.arange(len(competitors))
    
    # Create horizontal bar chart
    bars = ax.barh(y_pos, ratings, xerr=errors, capsize=3, 
                   color='#4285f4', edgecolor='#2c5282', linewidth=1,
                   error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': '#666666'})
    
    # Customize appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(competitors)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel(rating_name, fontsize=12)
    ax.set_ylabel(item_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add rating values on bars
    for i, (bar, rating) in enumerate(zip(bars, ratings)):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{rating:.1f}', va='center', fontsize=9, color='#333333')
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return True


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


@app.command()
def leaderboard(
    results_file: Path = typer.Option(
        Path("logs/eval/results.csv"),
        "--results", "-r",
        help="Path to results.csv file from Codenames tournament"
    ),
    top_n: int = typer.Option(20, "--top", "-n", help="Number of top models to show"),
    significance: float = typer.Option(0.05, "--significance", help="Significance level for confidence intervals"),
):
    """Generate Bradley-Terry leaderboard from tournament results.
    
    Uses the arena-rank library to compute ELO-style ratings with confidence
    intervals from head-to-head match results.
    
    See: https://github.com/lmarena/arena-rank
    """
    import pandas as pd
    
    try:
        from arena_rank.utils.data_utils import PairDataset
        from arena_rank.models.bradley_terry import BradleyTerry
    except ImportError:
        console.print("[red]Error: arena-rank not installed. Run: uv add arena-rank[/red]")
        raise typer.Exit(1)
    
    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        console.print("[dim]Run a tournament first: uv run based codenames eval --schedule <schedule.yml>[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]🏆 Bradley-Terry Leaderboard[/bold blue]")
    console.print(f"Loading results from: {results_file}")
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Validate required columns
    required_cols = ["model_a", "model_b", "winner"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        console.print(f"[red]Error: Missing required columns: {missing_cols}[/red]")
        console.print(f"[dim]Expected columns: {required_cols}[/dim]")
        raise typer.Exit(1)
    
    total_games = len(df)
    unique_models = set(df["model_a"].unique()) | set(df["model_b"].unique())
    
    # Count ties
    tie_count = len(df[df["winner"].isin(["tie", "both_bad"])])
    
    console.print(f"Total games: {total_games}")
    console.print(f"  - Decisive: {total_games - tie_count}")
    console.print(f"  - Ties: {tie_count} (counted as 0.5 for each model in BT ratings)")
    console.print(f"Unique models: {len(unique_models)}")
    console.print(f"Significance level: {significance}")
    
    # Create dataset and model
    dataset = PairDataset.from_pandas(df)
    model = BradleyTerry(n_competitors=len(dataset.competitors))
    
    # Compute ratings and confidence intervals
    console.print("\n[dim]Computing Bradley-Terry ratings...[/dim]")
    results = model.compute_ratings_and_cis(dataset, significance_level=significance)
    
    # Create leaderboard DataFrame
    leaderboard_df = pd.DataFrame(results).sort_values("ratings", ascending=False)
    
    # Display as rich table
    table = Table(title=f"Top {min(top_n, len(leaderboard_df))} Models by Bradley-Terry Rating")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("Rating", style="green", justify="right")
    table.add_column("CI Lower", style="dim", justify="right")
    table.add_column("CI Upper", style="dim", justify="right")
    table.add_column("Variance", style="dim", justify="right")
    
    for i, (_, row) in enumerate(leaderboard_df.head(top_n).iterrows(), 1):
        table.add_row(
            str(i),
            row["competitors"],
            f"{row['ratings']:.2f}",
            f"{row['rating_lower']:.2f}",
            f"{row['rating_upper']:.2f}",
            f"{row['variances']:.4f}",
        )
    
    console.print()
    console.print(table)
    
    # Show win rates
    console.print("\n[bold]Win Rates:[/bold]")
    console.print("[dim]Note: Ties are counted as 0.5 wins for each model in Bradley-Terry ratings[/dim]")
    
    win_table = Table(title="Model Win Rates")
    win_table.add_column("Model", style="cyan")
    win_table.add_column("Wins", justify="right")
    win_table.add_column("Ties", justify="right", style="yellow")
    win_table.add_column("Losses", justify="right")
    win_table.add_column("Games", justify="right")
    win_table.add_column("Win %", style="green", justify="right")
    
    # Calculate win rates (including ties)
    win_counts = {}
    loss_counts = {}
    tie_counts = {}
    
    for _, row in df.iterrows():
        model_a = row["model_a"]
        model_b = row["model_b"]
        winner = row["winner"]
        
        # Initialize if needed
        for m in [model_a, model_b]:
            if m not in win_counts:
                win_counts[m] = 0
                loss_counts[m] = 0
                tie_counts[m] = 0
        
        if winner == "model_a":
            win_counts[model_a] += 1
            loss_counts[model_b] += 1
        elif winner == "model_b":
            win_counts[model_b] += 1
            loss_counts[model_a] += 1
        elif winner in ("tie", "both_bad"):
            # Ties count for both models
            tie_counts[model_a] += 1
            tie_counts[model_b] += 1
    
    # Sort by win rate (counting ties as 0.5 wins, matching BT model)
    win_rates = []
    for model in unique_models:
        wins = win_counts.get(model, 0)
        losses = loss_counts.get(model, 0)
        ties = tie_counts.get(model, 0)
        total = wins + losses + ties
        # Win rate: (wins + 0.5*ties) / total to match BT model treatment
        rate = ((wins + 0.5 * ties) / total * 100) if total > 0 else 0
        win_rates.append((model, wins, ties, losses, total, rate))
    
    win_rates.sort(key=lambda x: x[5], reverse=True)
    
    for model, wins, ties, losses, total, rate in win_rates[:top_n]:
        win_table.add_row(
            model,
            str(wins),
            str(ties),
            str(losses),
            str(total),
            f"{rate:.1f}%",
        )
    
    console.print(win_table)
    
    # Save full leaderboard to CSV
    output_file = results_file.parent / "leaderboard.csv"
    leaderboard_df.to_csv(output_file, index=False)
    console.print(f"\n[green]✅ Full leaderboard saved to: {output_file}[/green]")
    
    # Generate chart
    chart_file = results_file.parent / "leaderboard.png"
    console.print(f"\n[dim]Generating leaderboard chart...[/dim]")
    
    if plot_leaderboard(
        results=results,
        output_path=chart_file,
        top_n=top_n,
        title="Codenames Tournament - Bradley-Terry Ratings",
        item_name="Model",
        rating_name="BT Rating",
    ):
        console.print(f"[green]✅ Leaderboard chart saved to: {chart_file}[/green]")
