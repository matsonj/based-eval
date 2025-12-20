"""Core game logic for Codenames (part of BASED eval)."""

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.table import Table

from codenames.player import AIPlayer, HumanPlayer
from codenames.utils.logging import (
    log_game_start, log_spymaster_clue, log_operative_guess, 
    log_game_end, log_box_score, log_turn_end_status, log_referee_rejection, log_referee_penalty,
    log_ai_call_metadata, format_turn_label, log_game_setup_metadata
)
# Import shared controllog SDK for unified analytics
from shared import controllog as cl

console = Console()
logger = logging.getLogger(__name__)


class CodenamesGame:
    """The main game class that manages a complete Codenames game."""

    # Version for tracking evaluation framework changes
    VERSION = "3.0.0"  # Added tournament scheduling, parallelism, and Bradley-Terry output
    
    BOARD_SIZE = 25
    STARTING_TEAM_AGENTS = 9  # Team that goes first gets 9
    SECOND_TEAM_AGENTS = 8    # Team that goes second gets 8
    BYSTANDERS = 7
    ASSASSINS = 1

    def __init__(
        self,
        words_file: str,
        red_player,
        blue_player,
        referee_player=None,
        red_spymaster_prompt: str = "",
        red_operative_prompt: str = "",
        blue_spymaster_prompt: str = "",
        blue_operative_prompt: str = "",
        referee_prompt: str = "",
        interactive_mode: Optional[str] = None,
    ):
        self.words_file = words_file
        self.red_player = red_player
        self.blue_player = blue_player
        self.referee_player = referee_player
        self.interactive_mode = interactive_mode
        self.prompt_files = {
            "red_spymaster": red_spymaster_prompt,
            "red_operative": red_operative_prompt,
            "blue_spymaster": blue_spymaster_prompt,
            "blue_operative": blue_operative_prompt,
            "referee": referee_prompt,
        }

        # Game state
        self.board: List[str] = []
        self.identities: Dict[str, str] = {}  # word -> identity
        self.revealed: Dict[str, bool] = {}  # word -> revealed status
        # Randomly choose which team starts first
        self.starting_team = random.choice(["red", "blue"])
        self.current_team = self.starting_team
        self.game_over = False
        self.winner: Optional[str] = None
        self.turn_count = 0

        # Track game statistics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.moves_log: List[Dict] = []
        self.clue_history: List[Dict] = []
        
        # Track costs
        self.total_cost: float = 0.0
        self.total_upstream_cost: float = 0.0
        
        # Generate unique game ID
        import uuid
        self.game_id = str(uuid.uuid4())[:8]
        
        # Controllog state (initialized by CLI)
        self._controllog_initialized = False
        self._run_id: Optional[str] = None
        self._task_id: Optional[str] = None

    def init_controllog(self, log_path: Path, run_id: str) -> None:
        """Initialize controllog SDK for unified analytics.
        
        Args:
            log_path: Directory for JSONL logs
            run_id: Unique identifier for this run
        """
        try:
            cl.init(project_id="codenames", log_dir=log_path)
            self._controllog_initialized = True
            self._run_id = run_id
            self._task_id = f"game:{self.game_id}"
            logger.info(f"Controllog initialized for game {self.game_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize controllog: {e}")
            self._controllog_initialized = False

    def _emit_state_move(self, from_state: str, to_state: str, payload: Optional[Dict] = None) -> None:
        """Emit a state transition event via controllog."""
        if not self._controllog_initialized:
            return
        try:
            cl.state_move(
                task_id=self._task_id,
                from_=from_state,
                to=to_state,
                project_id="codenames",
                agent_id="agent:codenames",
                run_id=self._run_id,
                payload=payload or {"game_id": self.game_id},
            )
        except Exception as e:
            logger.debug(f"Failed to emit state move: {e}")

    def _emit_model_events(
        self,
        player: AIPlayer,
        call_type: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost: Optional[float] = None,
        upstream_cost: Optional[float] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        """Emit model_prompt and model_completion events via controllog."""
        if not self._controllog_initialized:
            return
        try:
            exchange_id = cl.new_id()
            model_id = player.adapter.model_mappings.get(player.model_name, player.model_name)
            
            # Emit prompt event
            cl.model_prompt(
                task_id=self._task_id,
                agent_id=f"agent:codenames:{call_type}",
                run_id=self._run_id,
                project_id="codenames",
                provider="openrouter",
                model=model_id,
                prompt_tokens=prompt_tokens,
                payload={
                    "game_id": self.game_id,
                    "call_type": call_type,
                    "turn": format_turn_label(self.turn_count, self.current_team, self.starting_team),
                    "team": self.current_team,
                    **(payload or {}),
                },
                exchange_id=exchange_id,
            )
            
            # Emit completion event
            cl.model_completion(
                task_id=self._task_id,
                agent_id=f"agent:codenames:{call_type}",
                run_id=self._run_id,
                project_id="codenames",
                provider="openrouter",
                model=model_id,
                completion_tokens=completion_tokens,
                wall_ms=int(latency_ms),
                cost_money=cost,
                upstream_cost_money=upstream_cost,
                payload={
                    "game_id": self.game_id,
                    "call_type": call_type,
                    "turn": format_turn_label(self.turn_count, self.current_team, self.starting_team),
                    "team": self.current_team,
                    **(payload or {}),
                },
                exchange_id=exchange_id,
            )
        except Exception as e:
            logger.debug(f"Failed to emit model events: {e}")

    def load_words(self) -> List[str]:
        """Load words from YAML file."""
        try:
            with open(self.words_file, "r") as f:
                data = yaml.safe_load(f)
                words = data.get("names", [])  # Keep "names" key for backward compatibility
                if len(words) < self.BOARD_SIZE:
                    raise ValueError(
                        f"Need at least {self.BOARD_SIZE} words, got {len(words)}"
                    )
                return words
        except FileNotFoundError:
            logger.error(f"Words file not found: {self.words_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading words: {e}")
            raise

    def setup_board(self):
        """Initialize the game board with random word assignment."""
        all_words = self.load_words()

        # Select 25 random words
        self.board = random.sample(all_words, self.BOARD_SIZE)

        # Assign identities
        positions = list(range(self.BOARD_SIZE))
        random.shuffle(positions)

        # Assign agents based on who starts first
        if self.starting_team == "red":
            red_count = self.STARTING_TEAM_AGENTS
            blue_count = self.SECOND_TEAM_AGENTS
        else:
            red_count = self.SECOND_TEAM_AGENTS
            blue_count = self.STARTING_TEAM_AGENTS
        
        red_positions = positions[:red_count]
        blue_positions = positions[red_count:red_count + blue_count]

        # Assign assassin and bystanders
        remaining_positions = positions[red_count + blue_count:]
        assassin_position = remaining_positions[0]
        bystander_positions = remaining_positions[1 : 1 + self.BYSTANDERS]

        # Create identity mapping
        self.identities = {}
        self.revealed = {}

        for i, word in enumerate(self.board):
            if i in red_positions:
                self.identities[word] = "red_agent"
            elif i in blue_positions:
                self.identities[word] = "blue_agent"
            elif i == assassin_position:
                self.identities[word] = "assassin"
            else:
                self.identities[word] = "bystander"

            self.revealed[word] = False

        logger.info(
            f"Board setup complete. Starting team: {self.starting_team.upper()}. Red: {len(red_positions)}, Blue: {len(blue_positions)}, Bystanders: {len(bystander_positions)}, Assassin: 1"
        )

    def get_board_state(self, reveal_all: bool = False) -> Dict[str, Any]:
        """Get current board state for display."""
        identities: Dict[str, str] = {} if not reveal_all else self.identities.copy()

        # Add revealed identities
        if not reveal_all:
            for word in self.board:
                if self.revealed.get(word, False):
                    identities[word] = self.identities[word]

        state = {
            "board": self.board.copy(),
            "revealed": self.revealed.copy(),
            "identities": identities,
            "current_team": self.current_team,
            "turn_count": self.turn_count,
            "clue_history": self.format_clue_history(),
        }

        return state

    def _format_board_for_operative_cli(self, board_state: dict) -> str:
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

    def display_board_start(self):
        """Display the initial board state at game start."""
        console.print(f"\n[bold]Game Board - {self.current_team.title()} Team Goes First[/bold]")
        
        # Create a 5x5 grid
        table = Table(show_header=False, show_lines=True)
        for _ in range(5):
            table.add_column(justify="center", min_width=12)

        for row in range(5):
            row_items = []
            for col in range(5):
                idx = row * 5 + col
                word = self.board[idx]
                row_items.append(f"[white]{word}[/white]")
            table.add_row(*row_items)

        console.print(table)
        
        # Show team info
        red_total = sum(1 for identity in self.identities.values() if identity == "red_agent")
        blue_total = sum(1 for identity in self.identities.values() if identity == "blue_agent")
        bystander_total = sum(1 for identity in self.identities.values() if identity == "bystander")
        
        console.print(f"\n[red]Red Team:[/red] {red_total} agents")
        console.print(f"[blue]Blue Team:[/blue] {blue_total} agents")
        console.print(f"[dim]Bystanders:[/dim] {bystander_total}")
        console.print(f"[black on white]Assassin:[/black on white] 1")
        console.print("")

    def display_board(self, reveal_all: bool = False):
        """Display the current board state."""
        state = self.get_board_state(reveal_all)

        console.print(
            f"\n[bold]Turn {self.turn_count + 1} - {self.current_team.title()} Team[/bold]"
        )

        # Create a 5x5 grid
        table = Table(show_header=False, show_lines=True)
        for _ in range(5):
            table.add_column()

        for row in range(5):
            row_items = []
            for col in range(5):
                idx = row * 5 + col
                word = state["board"][idx]

                # Color coding based on identity (if revealed or reveal_all)
                if word in state["identities"]:
                    identity = state["identities"][word]
                    if identity == "red_agent":
                        color = "red"
                    elif identity == "blue_agent":
                        color = "blue"
                    elif identity == "assassin":
                        color = "black on white"
                    else:  # bystander
                        color = "dim"
                else:
                    color = "white"

                # Add revealed indicator
                display_word = word
                if self.revealed[word]:
                    display_word = f"[{word}]"

                row_items.append(f"[{color}]{display_word}[/{color}]")

            table.add_row(*row_items)

        console.print(table)

        # Show team counts
        red_remaining = sum(
            1
            for word, identity in self.identities.items()
            if identity == "red_agent" and not self.revealed[word]
        )
        blue_remaining = sum(
            1
            for word, identity in self.identities.items()
            if identity == "blue_agent" and not self.revealed[word]
        )

        console.print(
            f"\n[red]Red Team Remaining: {red_remaining}[/red]  [blue]Blue Team Remaining: {blue_remaining}[/blue]"
        )

    def _format_board_for_operative_prompt(self, board_state: Dict) -> str:
        """Format the board for operative prompt display with revealed status."""
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

    def get_spymaster_turn(self) -> Tuple[Optional[str], Optional[int|str]]:
        """Get clue and number from the current team's spymaster."""
        player = self.red_player if self.current_team == "red" else self.blue_player
        prompt_key = f"{self.current_team}_spymaster"

        # Check if this specific role should be human
        is_human_spymaster = (self.interactive_mode == f"{self.current_team}-spymaster")
        
        if is_human_spymaster:
            # Display the spymaster prompt first
            board_state = self.get_board_state(reveal_all=True)
            from codenames.prompt_manager import PromptManager
            prompt_manager = PromptManager()
            
            # Calculate remaining agents
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
            
            prompt = prompt_manager.load_prompt(
                self.prompt_files[prompt_key],
                {
                    "board": board_state["board"],
                    "revealed": board_state["revealed"],
                    "team": self.current_team,
                    "red_remaining": red_remaining,
                    "blue_remaining": blue_remaining,
                    "revealed_words": ", ".join(revealed_words) if revealed_words else "None",
                    "red_agents": ", ".join(red_agents),
                    "blue_agents": ", ".join(blue_agents),
                    "bystanders": ", ".join(bystanders),
                    "assassin": ", ".join(assassin),
                },
            )
            
            console.print(f"\n[bold]{self.current_team.title()} Spymaster Turn (Human)[/bold]")
            console.print(f"[yellow]{'='*80}[/yellow]")
            console.print("[yellow]SPYMASTER PROMPT:[/yellow]")
            console.print(f"[yellow]{'='*80}[/yellow]")
            console.print(prompt)
            console.print(f"[yellow]{'='*80}[/yellow]\n")

            clue = console.input("Enter your clue: ").strip()
            number: int|str
            while True:
                try:
                    number_input = console.input("Enter number of related words (or 'unlimited'): ").strip().lower()
                    if number_input == "unlimited":
                        number = "unlimited"
                        break
                    else:
                        number_val = int(number_input)
                        if number_val >= 0:
                            number = number_val
                            break
                        console.print("[red]Number must be 0 or positive[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid number or 'unlimited'[/red]")

            # Validate clue with referee if available
            if self.referee_player:
                board_state = self.get_board_state(reveal_all=True)
                validated_clue, validated_number, is_valid, reasoning = self._validate_clue_with_referee(clue, number, board_state)
                if not is_valid:
                    # Record invalid clue in history for future reference
                    self.record_clue(self.current_team, clue, number, invalid=True, invalid_reason=reasoning)
                    # Log the rejected clue and end turn
                    log_spymaster_clue(self.current_team, "human", f"REJECTED: {clue}", number, self.turn_count, self.starting_team)
                    return None, None  # Signal that turn should end
            
            # Log the clue
            log_spymaster_clue(self.current_team, "human", clue, number, self.turn_count, self.starting_team)
            return clue, number

        else:  # AI Player
            board_state = self.get_board_state(reveal_all=True)
            clue, number = player.get_spymaster_move(
                board_state, self.prompt_files[prompt_key]
            )
            console.print(
                f'[{self.current_team}]{self.current_team.title()} Spymaster[/{self.current_team}]: "{clue}" ({number})'
            )
            
            # Log AI call metadata first (before referee validation) if this is an AI player
            if isinstance(player, AIPlayer):
                metadata = player.get_last_call_metadata()
                if metadata:
                    # Accumulate costs
                    self.total_cost += metadata.get("openrouter_cost", 0.0)
                    self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                    
                    turn_label = format_turn_label(self.turn_count, self.current_team, self.starting_team)
                    log_ai_call_metadata(
                        game_id=self.game_id,
                        model_name=player.model_name,
                        call_type=metadata["call_type"],
                        team=self.current_team,
                        turn=turn_label,
                        input_tokens=metadata["input_tokens"],
                        output_tokens=metadata["output_tokens"],
                        total_tokens=metadata["total_tokens"],
                        latency_ms=metadata["latency_ms"],
                        openrouter_cost=metadata.get("openrouter_cost", 0.0),
                        upstream_cost=metadata.get("upstream_cost", 0.0),
                        turn_result=metadata.get("turn_result", {}),
                        game_continues=not self.game_over,
                        is_retry=metadata.get("is_retry", False)
                    )
                    # Emit controllog model events for spymaster
                    self._emit_model_events(
                        player=player,
                        call_type="spymaster",
                        prompt_tokens=metadata["input_tokens"],
                        completion_tokens=metadata["output_tokens"],
                        latency_ms=metadata["latency_ms"],
                        cost=metadata.get("openrouter_cost"),
                        upstream_cost=metadata.get("upstream_cost"),
                        payload={
                            "clue": clue,
                            "number": number if isinstance(number, (int, str)) else str(number),
                            "is_retry": metadata.get("is_retry", False),
                        },
                    )
            
            # Validate clue with referee if available
            if self.referee_player:
                validated_clue, validated_number, is_valid, reasoning = self._validate_clue_with_referee(clue, number, board_state)
                if not is_valid:
                    # Record invalid clue in history for future reference
                    self.record_clue(self.current_team, clue, number, invalid=True, invalid_reason=reasoning)
                    # Log the rejected clue and end turn
                    log_spymaster_clue(self.current_team, player.model_name, f"REJECTED: {clue}", number, self.turn_count, self.starting_team)
                    return None, None  # Signal that turn should end
            
            # Log the clue
            log_spymaster_clue(self.current_team, player.model_name, clue, number, self.turn_count, self.starting_team)
            
            return clue, number

    def get_operative_guesses(self, clue: str, number: int|str) -> List[str]:
        """Get guesses from the current team's operatives."""
        player = self.red_player if self.current_team == "red" else self.blue_player
        prompt_key = f"{self.current_team}_operative"

        # Check if this specific role should be human
        is_human_operative = (self.interactive_mode == f"{self.current_team}-operative")
        
        if is_human_operative:
            # Display the operative prompt first
            board_state = self.get_board_state(reveal_all=False)
            from codenames.prompt_manager import PromptManager
            prompt_manager = PromptManager()
            
            # Filter board to only show available (unrevealed) words
            available_words = [
                word for word in board_state["board"] 
                if not board_state["revealed"].get(word, False)
            ]
            
            # Format available words as a simple list
            available_words_formatted = ", ".join(available_words)
            
            prompt = prompt_manager.load_prompt(
                self.prompt_files[prompt_key],
                {
                    "BOARD": self._format_board_for_operative_prompt(board_state),
                    "AVAILABLE_WORDS": available_words_formatted,
                    "CLUE_HISTORY": board_state.get("clue_history", "None (game just started)"),
                    "CLUE": clue,
                    "NUMBER": number,
                    "TEAM": self.current_team,
                },
            )
            
            console.print(f"\n[bold]{self.current_team.title()} Operative Turn (Human)[/bold]")
            console.print(f"[yellow]{'='*80}[/yellow]")
            console.print("[yellow]OPERATIVE PROMPT:[/yellow]")
            console.print(f"[yellow]{'='*80}[/yellow]")
            console.print(prompt)
            console.print(f"[yellow]{'='*80}[/yellow]\n")

            guesses: List[str] = []
            
            # Determine max guesses based on clue type
            if number == "unlimited" or number == 0:
                max_guesses = len([word for word in self.board if not self.revealed[word]])  # All available words
                min_guesses = 1 if number == 0 else 0  # Zero clues require at least one guess
            elif isinstance(number, int):
                max_guesses = number + 1  # Plus-one rule
                min_guesses = 0
            else:
                max_guesses = 1  # Fallback
                min_guesses = 0

            for i in range(max_guesses):
                available_words = [
                    word for word in self.board if not self.revealed[word]
                ]

                console.print(f"\nAvailable words: {', '.join(available_words)}")
                
                # Show appropriate prompt based on clue type
                if number == "unlimited":
                    prompt = f"Guess {i+1} (or 'done' to stop): "
                elif number == 0:
                    if i == 0:
                        prompt = f"Guess {i+1} (required for zero clue): "
                    else:
                        prompt = f"Guess {i+1} (or 'done' to stop): "
                else:
                    prompt = f"Guess {i+1}/{max_guesses} (or 'done' to stop): "
                
                guess = console.input(prompt).strip()

                if guess.lower() == "done":
                    # Check minimum guess requirement for zero clues
                    if number == 0 and len(guesses) == 0:
                        console.print(f"[red]Zero clues require at least one guess[/red]")
                        continue
                    break

                if guess not in available_words:
                    console.print(f"[red]'{guess}' is not available. Try again.[/red]")
                    continue

                guesses.append(guess)

                # Process guess immediately
                result = self.process_guess(guess)
                if not result:  # Wrong guess ends turn
                    break

            return guesses

        else:  # AI Player
            board_state = self.get_board_state(reveal_all=False)
            guesses = player.get_operative_moves(
                board_state, clue, number, self.prompt_files[prompt_key]
            )

            # Track guess results for metadata logging
            guess_results = []
            
            # Process guesses one by one
            for guess in guesses:
                console.print(
                    f"[{self.current_team}]{self.current_team.title()} Operative[/{self.current_team}] guesses: {guess}"
                )
                result = self.process_guess(guess)
                
                # Track result for metadata
                if guess in self.identities:
                    identity = self.identities[guess]
                    if identity == f"{self.current_team}_agent":
                        guess_results.append({"guess": guess, "result": "correct"})
                    elif identity == "assassin":
                        guess_results.append({"guess": guess, "result": "assassin"})
                    elif identity == "bystander":
                        guess_results.append({"guess": guess, "result": "bystander"})
                    else:  # enemy agent
                        guess_results.append({"guess": guess, "result": "enemy"})
                
                if not result:  # Wrong guess ends turn
                    break

            # Log AI call metadata if this is an AI player
            if isinstance(player, AIPlayer):
                metadata = player.get_last_call_metadata()
                if metadata:
                    # Accumulate costs
                    self.total_cost += metadata.get("openrouter_cost", 0.0)
                    self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                    
                    turn_label = format_turn_label(self.turn_count, self.current_team, self.starting_team)
                    
                    # Add detailed results from processing guesses
                    turn_result = metadata.get("turn_result", {})
                    turn_result.update({
                        "correct_guesses": sum(1 for r in guess_results if r["result"] == "correct"),
                        "bystander_hits": sum(1 for r in guess_results if r["result"] == "bystander"),
                        "enemy_hits": sum(1 for r in guess_results if r["result"] == "enemy"),
                        "assassin_hits": sum(1 for r in guess_results if r["result"] == "assassin"),
                        "guess_details": guess_results
                    })
                    
                    log_ai_call_metadata(
                        game_id=self.game_id,
                        model_name=player.model_name,
                        call_type=metadata["call_type"],
                        team=self.current_team,
                        turn=turn_label,
                        input_tokens=metadata["input_tokens"],
                        output_tokens=metadata["output_tokens"],
                        total_tokens=metadata["total_tokens"],
                        latency_ms=metadata["latency_ms"],
                        openrouter_cost=metadata.get("openrouter_cost", 0.0),
                        upstream_cost=metadata.get("upstream_cost", 0.0),
                        turn_result=turn_result,
                        game_continues=not self.game_over,
                        is_retry=metadata.get("is_retry", False)
                    )
                    # Emit controllog model events for operative
                    self._emit_model_events(
                        player=player,
                        call_type="operative",
                        prompt_tokens=metadata["input_tokens"],
                        completion_tokens=metadata["output_tokens"],
                        latency_ms=metadata["latency_ms"],
                        cost=metadata.get("openrouter_cost"),
                        upstream_cost=metadata.get("upstream_cost"),
                        payload={
                            "clue_given": clue,
                            "clue_number": number if isinstance(number, (int, str)) else str(number),
                            "guesses": guesses,
                            "correct_guesses": turn_result.get("correct_guesses", 0),
                            "is_retry": metadata.get("is_retry", False),
                        },
                    )

            return guesses

    def process_guess(self, word: str) -> bool:
        """Process a single guess and return True if correct, False if wrong."""
        if word not in self.identities:
            logger.warning(f"Invalid guess: {word}")
            return False

        identity = self.identities[word]
        self.revealed[word] = True

        # Log the move
        move = {
            "team": self.current_team,
            "word": word,
            "identity": identity,
            "correct": identity == f"{self.current_team}_agent",
        }
        self.moves_log.append(move)

        # Record guess outcome for clue history
        correct = identity == f"{self.current_team}_agent"
        self.record_guess_outcome(word, identity, correct)

        # Determine result type for logging
        player = self.red_player if self.current_team == "red" else self.blue_player
        model_name = player.model_name if hasattr(player, 'model_name') else "human"

        if identity == "assassin":
            console.print(
                f"[black on white]💀 THE ASSASSIN! {self.current_team.title()} team loses![/black on white]"
            )
            log_operative_guess(self.current_team, model_name, word, "assassin", self.turn_count, self.starting_team)
            self.game_over = True
            self.winner = "blue" if self.current_team == "red" else "red"
            return False

        elif identity == f"{self.current_team}_agent":
            console.print(f"[green]✓ Correct! {word} is one of your agents![/green]")
            log_operative_guess(self.current_team, model_name, word, "correct", self.turn_count, self.starting_team)

            # Check win condition
            remaining = sum(
                1
                for w, i in self.identities.items()
                if i == f"{self.current_team}_agent" and not self.revealed[w]
            )
            if remaining == 0:
                console.print(
                    f"[green]🎉 {self.current_team.title()} team wins![/green]"
                )
                self.game_over = True
                self.winner = self.current_team

            return True

        else:
            if identity == "bystander":
                console.print(f"[yellow]✗ {word} is an innocent bystander.[/yellow]")
                log_operative_guess(self.current_team, model_name, word, "bystander", self.turn_count, self.starting_team)
            else:
                console.print(f"[dim]✗ {word} is an enemy agent![/dim]")
                log_operative_guess(self.current_team, model_name, word, "enemy", self.turn_count, self.starting_team)
                
                # Check if the opposing team just won by having this team hit their agent
                opposing_team = "blue" if self.current_team == "red" else "red"
                remaining = sum(
                    1
                    for w, i in self.identities.items()
                    if i == f"{opposing_team}_agent" and not self.revealed[w]
                )
                if remaining == 0:
                    console.print(
                        f"[green]🎉 {opposing_team.title()} team wins![/green]"
                    )
                    self.game_over = True
                    self.winner = opposing_team
                    
            return False

    def get_remaining_agents(self):
        """Get remaining agent counts for both teams."""
        red_remaining = sum(
            1 for word, identity in self.identities.items()
            if identity == "red_agent" and not self.revealed[word]
        )
        blue_remaining = sum(
            1 for word, identity in self.identities.items()
            if identity == "blue_agent" and not self.revealed[word]
        )
        return red_remaining, blue_remaining

    def display_game_status(self):
        """Display the current game status showing remaining agents."""
        red_remaining, blue_remaining = self.get_remaining_agents()
        
        # Always show starting team first
        if self.starting_team == "red":
            console.print(f"[bold]Status:[/bold] [red]Red {red_remaining}[/red], [blue]Blue {blue_remaining}[/blue]")
        else:
            console.print(f"[bold]Status:[/bold] [blue]Blue {blue_remaining}[/blue], [red]Red {red_remaining}[/red]")
        console.print("")

    def record_clue(self, team: str, clue: str, number: int|str, invalid: bool = False, invalid_reason: str = ""):
        """Record a clue for the game history."""
        clue_entry = {
            "turn": self.turn_count,
            "team": team,
            "clue": clue,
            "number": number,
            "guesses": [],
            "invalid": invalid,
            "invalid_reason": invalid_reason
        }
        self.clue_history.append(clue_entry)

    def record_guess_outcome(self, word: str, identity: str, correct: bool):
        """Record the outcome of a guess for the current clue."""
        if self.clue_history:
            current_clue = self.clue_history[-1]
            outcome = "correct" if correct else ("enemy" if identity.endswith("_agent") else ("bystander" if identity == "bystander" else "assassin"))
            current_clue["guesses"].append({
                "word": word,
                "identity": identity,
                "outcome": outcome
            })

    def format_clue_history(self) -> str:
        """Format the clue history for display to operatives."""
        if not self.clue_history:
            return "None (game just started)"
        
        history_lines = []
        for entry in self.clue_history:
            turn_letter = "a" if entry["team"] == self.starting_team else "b"
            turn_label = f"Turn {entry['turn'] + 1}{turn_letter}"
            
            # Format the clue line
            if entry.get("invalid", False):
                clue_line = f"{turn_label}: {entry['team'].title()} Clue: \"{entry['clue']}\" ({entry['number']}) [INVALID: {entry.get('invalid_reason', 'rule violation')}]"
            else:
                clue_line = f"{turn_label}: {entry['team'].title()} Clue: \"{entry['clue']}\" ({entry['number']})"
            history_lines.append(clue_line)
            
            # Format the outcomes
            if entry.get("invalid", False):
                history_lines.append("  → Turn ended due to invalid clue")
            elif entry["guesses"]:
                outcomes = []
                for guess in entry["guesses"]:
                    if guess["outcome"] == "correct":
                        outcomes.append(f"{guess['word']} ✓")
                    elif guess["outcome"] == "enemy":
                        outcomes.append(f"{guess['word']} ✗ (enemy)")
                    elif guess["outcome"] == "bystander":
                        outcomes.append(f"{guess['word']} ○ (bystander)")
                    # Note: assassin outcomes end the game, so we don't need to handle them here
                
                if outcomes:
                    history_lines.append(f"  → {', '.join(outcomes)}")
            else:
                history_lines.append("  → No guesses made")
            
            history_lines.append("")  # Empty line for spacing
        
        return "\n".join(history_lines).strip()

    def _validate_clue_with_referee(self, clue: str, number: int|str, board_state: Dict) -> Tuple[str, int|str, bool, str]:
        """Validate clue with referee and handle invalid clues. Returns (clue, number, is_valid, reasoning)."""
        try:
            if self.interactive_mode == "referee":
                # Human referee validation
                from codenames.prompt_manager import PromptManager
                prompt_manager = PromptManager()
                
                # Get team's agents
                team_agents = [
                    word for word, identity in board_state["identities"].items()
                    if identity == f"{self.current_team}_agent"
                ]
                
                prompt = prompt_manager.load_prompt(
                    self.prompt_files["referee"],
                    {
                        "clue": clue,
                        "number": number,
                        "team": self.current_team,
                        "board": board_state["board"],
                        "team_agents": ", ".join(team_agents),
                    },
                )
                
                console.print(f"\n[bold]Referee Validation (Human)[/bold]")
                console.print(f"Team: {self.current_team.title()}")
                console.print(f'Clue: "{clue}" ({number})')
                console.print(f"[yellow]{'='*80}[/yellow]")
                console.print("[yellow]REFEREE PROMPT:[/yellow]")
                console.print(f"[yellow]{'='*80}[/yellow]")
                console.print(prompt)
                console.print(f"[yellow]{'='*80}[/yellow]\n")
                
                while True:
                    decision = console.input("Is this clue valid? (y/n): ").strip().lower()
                    if decision in ['y', 'yes']:
                        reasoning = console.input("Reasoning (optional): ").strip() or "Clue approved by human referee"
                        is_valid = True
                        break
                    elif decision in ['n', 'no']:
                        reasoning = console.input("Violation reasoning: ").strip() or "Rule violation detected by human referee"
                        is_valid = False
                        break
                    else:
                        console.print("[red]Please enter 'y' or 'n'[/red]")
            else:
                # AI referee validation
                is_valid, reasoning = self.referee_player.get_referee_validation(
                    clue, number, self.current_team, board_state, self.prompt_files["referee"]
                )
                
                # If first referee flags as invalid, do second review with gpt5.2
                if not is_valid and self.referee_player is not None:
                    console.print(f"[yellow]🔄 First referee flagged clue as invalid. Getting second opinion from gpt5.2...[/yellow]")
                    
                    # Create a temporary gpt5.2 player for second review (different model to avoid agreement bias)
                    review_referee = AIPlayer("gpt5.2")
                    
                    # Get second opinion with same prompt
                    review_valid, review_reasoning = review_referee.get_referee_validation(
                        clue, number, self.current_team, board_state, self.prompt_files["referee"]
                    )
                    
                    # Log the review referee metadata
                    review_metadata = review_referee.get_last_call_metadata()
                    if review_metadata:
                        # Accumulate costs for review referee
                        self.total_cost += review_metadata.get("openrouter_cost", 0.0)
                        self.total_upstream_cost += review_metadata.get("upstream_cost", 0.0)
                        
                        turn_label = format_turn_label(self.turn_count, self.current_team, self.starting_team)
                        
                        # Update turn result with review referee validation outcome
                        turn_result = review_metadata.get("turn_result", {})
                        turn_result.update({
                            "evaluated_clue": clue,
                            "evaluated_number": number,
                            "review_referee": True,
                            "first_referee_model": self.referee_player.model_name,
                            "first_referee_decision": "invalid",
                            "first_referee_reasoning": reasoning
                        })
                        
                        log_ai_call_metadata(
                            game_id=self.game_id,
                            model_name=review_referee.model_name,
                            call_type=review_metadata["call_type"],
                            team=f"review_referee_{self.current_team}",
                            turn=turn_label,
                            input_tokens=review_metadata["input_tokens"],
                            output_tokens=review_metadata["output_tokens"],
                            total_tokens=review_metadata["total_tokens"],
                            latency_ms=review_metadata["latency_ms"],
                            openrouter_cost=review_metadata.get("openrouter_cost", 0.0),
                            upstream_cost=review_metadata.get("upstream_cost", 0.0),
                            turn_result=turn_result,
                            game_continues=not self.game_over,
                            is_retry=review_metadata.get("is_retry", False)
                        )
                        # Emit controllog model events for review referee
                        self._emit_model_events(
                            player=review_referee,
                            call_type="review_referee",
                            prompt_tokens=review_metadata["input_tokens"],
                            completion_tokens=review_metadata["output_tokens"],
                            latency_ms=review_metadata["latency_ms"],
                            cost=review_metadata.get("openrouter_cost"),
                            upstream_cost=review_metadata.get("upstream_cost"),
                            payload={
                                "evaluated_clue": clue,
                                "evaluated_number": number if isinstance(number, (int, str)) else str(number),
                                "review_decision": "valid" if review_valid else "invalid",
                            },
                        )
                    
                    if review_valid:
                        # Second referee says it's valid - override first decision
                        console.print(f"[green]✅ The ruling is overturned![/green]")
                        console.print(f"[dim]First referee ({self.referee_player.model_name}) - {reasoning}[/dim]")
                        console.print(f"[dim]Review referee: {review_reasoning}[/dim]")
                        is_valid = True
                        reasoning = f"Approved on review by Gemini 2.5 Pro: {review_reasoning}"
                    else:
                        # Both referees say invalid - reject the clue
                        console.print(f"[yellow]❌ The ruling on the clue stands![/yellow]")
                        console.print(f"[dim]First referee ({self.referee_player.model_name}): {reasoning}[/dim]")
                        console.print(f"[dim]Review referee: {review_reasoning}[/dim]")
                        reasoning = f"Upheld on review. First: {reasoning}. Review: {review_reasoning}"
            
            # Log AI call metadata for referee validation
            if isinstance(self.referee_player, AIPlayer):
                metadata = self.referee_player.get_last_call_metadata()
                if metadata:
                    # Accumulate costs for referee
                    self.total_cost += metadata.get("openrouter_cost", 0.0)
                    self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                    
                    turn_label = format_turn_label(self.turn_count, self.current_team, self.starting_team)
                    
                    # Update turn result with referee validation outcome
                    turn_result = metadata.get("turn_result", {})
                    turn_result.update({
                        "evaluated_clue": clue,
                        "evaluated_number": number
                    })
                    
                    log_ai_call_metadata(
                        game_id=self.game_id,
                        model_name=self.referee_player.model_name,
                        call_type=metadata["call_type"],
                        team=f"referee_{self.current_team}",  # Include which team's clue was evaluated
                        turn=turn_label,
                        input_tokens=metadata["input_tokens"],
                        output_tokens=metadata["output_tokens"],
                        total_tokens=metadata["total_tokens"],
                        latency_ms=metadata["latency_ms"],
                        openrouter_cost=metadata.get("openrouter_cost", 0.0),
                        upstream_cost=metadata.get("upstream_cost", 0.0),
                        turn_result=turn_result,
                        game_continues=not self.game_over,
                        is_retry=metadata.get("is_retry", False)
                    )
                    # Emit controllog model events for referee
                    self._emit_model_events(
                        player=self.referee_player,
                        call_type="referee",
                        prompt_tokens=metadata["input_tokens"],
                        completion_tokens=metadata["output_tokens"],
                        latency_ms=metadata["latency_ms"],
                        cost=metadata.get("openrouter_cost"),
                        upstream_cost=metadata.get("upstream_cost"),
                        payload={
                            "evaluated_clue": clue,
                            "evaluated_number": number if isinstance(number, (int, str)) else str(number),
                            "referee_decision": "valid" if is_valid else "invalid",
                            "referee_reasoning": reasoning,
                        },
                    )
            
            if is_valid:
                return clue, number, True, reasoning
            else:
                console.print(f"⚠️  Turn ended due to invalid clue")
                log_referee_rejection(self.current_team, clue, number, reasoning)
                return clue, number, False, reasoning
                
        except Exception as e:
            logger.error(f"Error in referee validation: {e}")
            console.print(f"[yellow]⚠️  Referee error, allowing original clue[/yellow]")
            return clue, number, True, "Referee error - clue allowed"

    def apply_invalid_clue_penalty(self):
        """Apply penalty for invalid clue: reveal one of the opposing team's agents."""
        # Get opposing team
        opposing_team = "blue" if self.current_team == "red" else "red"
        
        # Find unrevealed opposing team agents
        opposing_agents = [
            word for word, identity in self.identities.items()
            if identity == f"{opposing_team}_agent" and not self.revealed[word]
        ]
        
        if opposing_agents:
            # Randomly select one to reveal
            penalty_word = random.choice(opposing_agents)
            self.revealed[penalty_word] = True
            
            console.print(f"[dim]⚖️  PENALTY: {penalty_word} revealed for {opposing_team.upper()} team due to invalid clue[/dim]")
            
            # Log the penalty action
            log_referee_penalty(self.current_team, opposing_team, penalty_word)
            
            logger.info(f"Invalid clue penalty applied: revealed {penalty_word} for {opposing_team} team")
            
            # Check if the violating team wins (opposing team has no agents left)
            remaining_opposing_agents = sum(
                1
                for word, identity in self.identities.items()
                if identity == f"{opposing_team}_agent" and not self.revealed[word]
            )
            
            if remaining_opposing_agents == 0:
                console.print(
                    f"[green]🎉 {self.current_team.title()} team wins![/green]"
                )
                self.game_over = True
                self.winner = self.current_team
            
            return penalty_word
        else:
            console.print(f"[yellow]⚖️  PENALTY: No unrevealed {opposing_team.upper()} agents to reveal[/yellow]")
            logger.info(f"Invalid clue penalty: no unrevealed {opposing_team} agents available")
            return None

    def switch_teams(self):
        """Switch to the other team."""
        # Log status before switching
        red_remaining, blue_remaining = self.get_remaining_agents()
        log_turn_end_status(red_remaining, blue_remaining)
        
        # Display game status to terminal
        self.display_game_status()
        
        self.current_team = "blue" if self.current_team == "red" else "red"
        self.turn_count += 1

    def play(self) -> Dict:
        """Play a complete game and return results."""
        self.start_time = time.time()

        logger.info("Starting new Codenames game")
        self.setup_board()
        
        # Log game start
        red_model = self.red_player.model_name if hasattr(self.red_player, 'model_name') else "human"
        blue_model = self.blue_player.model_name if hasattr(self.blue_player, 'model_name') else "human"
        log_game_start(self.game_id, red_model, blue_model, self.board, self.identities)
        
        # Log game setup metadata
        log_game_setup_metadata(self.game_id, red_model, blue_model, self.prompt_files, self.board, self.identities)
        
        # Emit controllog state transition: NEW -> WIP
        self._emit_state_move("NEW", "WIP", {
            "game_id": self.game_id,
            "red_model": red_model,
            "blue_model": blue_model,
            "starting_team": self.starting_team,
        })

        console.print("[bold]🎯 Codenames Game Starting![/bold]")
        console.print(f"[red]Red Team:[/red] {red_model}")
        console.print(f"  • Spymaster: {self.prompt_files.get('red_spymaster', 'default')}")
        console.print(f"  • Operative: {self.prompt_files.get('red_operative', 'default')}")
        console.print(f"[blue]Blue Team:[/blue] {blue_model}")
        console.print(f"  • Spymaster: {self.prompt_files.get('blue_spymaster', 'default')}")
        console.print(f"  • Operative: {self.prompt_files.get('blue_operative', 'default')}")
        if self.referee_player:
            referee_model = self.referee_player.model_name if hasattr(self.referee_player, 'model_name') else "human"
            console.print(f"[yellow]Referee:[/yellow] {referee_model} ({self.prompt_files.get('referee', 'default')})")
        else:
            console.print("[yellow]Referee:[/yellow] Disabled")
        console.print(f"[green]Game ID:[/green] {self.game_id}")
        console.print()
        
        # Display the initial board
        self.display_board_start()

        while not self.game_over:
            # Spymaster phase
            clue, number = self.get_spymaster_turn()
            
            # Check if clue was rejected by referee
            if clue is None or number is None:
                # Clue was rejected, apply penalty and end turn immediately
                self.apply_invalid_clue_penalty()
                self.switch_teams()
                continue

            # Record the clue for history tracking
            self.record_clue(self.current_team, clue, number)

            # Operative phase - clue and number are guaranteed to be non-None at this point
            guesses = self.get_operative_guesses(clue, number)

            if not self.game_over:
                self.switch_teams()

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Compile results
        result = {
            "winner": self.winner,
            "turns": self.turn_count,
            "duration": duration,
            "moves": self.moves_log,
            "final_board": self.get_board_state(reveal_all=True),
        }

        # Log game end and box score
        log_game_end(self.winner or "draw", self.turn_count, duration)
        log_box_score(self.game_id, red_model, blue_model, result)
        
        # Emit controllog state transition: WIP -> DONE or WIP -> FAILED
        final_state = "DONE" if self.winner else "FAILED"
        self._emit_state_move("WIP", final_state, {
            "game_id": self.game_id,
            "winner": self.winner,
            "turns": self.turn_count,
            "duration_sec": duration,
        })

        logger.info(f"Game completed. Winner: {self.winner}, Turns: {self.turn_count}")
        return result


# Backward compatibility alias
PlaybookGame = CodenamesGame
