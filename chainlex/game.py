"""Core game logic for ChainLex-1."""

import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.table import Table

from chainlex.player import AIPlayer
from shared import controllog as cl

console = Console()
logger = logging.getLogger(__name__)

# Cache for loaded word lists (keyed by file path)
_WORDS_CACHE: Dict[str, List[str]] = {}


class ChainLexGame:
    """The main game class for ChainLex-1.
    
    ChainLex-1 is a head-to-head word association game where:
    - Board has 16 words: 8 friendly, 7 bystanders, 1 assassin
    - TWO players compete on the SAME board
    - Each player gives ONE clue and then guesses (independently)
    - Scoring: nth correct guess = n points (triangular: 1+2+3+...)
    - Bystander: -5 points (ends turn)
    - Assassin: -28 points (ends turn)
    - Winner: Higher score on the same board
    """

    # Version for tracking evaluation framework changes
    VERSION = "1.0.0"
    
    # Board configuration
    BOARD_SIZE = 16
    FRIENDLY_WORDS = 8
    BYSTANDERS = 7
    ASSASSINS = 1
    
    # Scoring
    BYSTANDER_PENALTY = -5
    ASSASSIN_PENALTY = -28

    def __init__(
        self,
        words_file: str,
        player_away,
        player_home,
        clue_giver_prompt: str = "",
        guesser_prompt: str = "",
        quiet: bool = False,
        seed: Optional[int] = None,
    ):
        self.words_file = words_file
        self.player_away = player_away  # Goes first
        self.player_home = player_home  # Goes second (knows opponent's score)
        self.quiet = quiet
        self.seed = seed
        
        self.prompt_files = {
            "clue_giver": clue_giver_prompt,
            "guesser": guesser_prompt,
        }

        # Game state
        self.board: List[str] = []
        self.identities: Dict[str, str] = {}  # word -> identity
        
        # Per-player state (each player gets fresh revealed state)
        self.player_results: Dict[str, Dict] = {}

        # Track game statistics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Track costs
        self.total_cost: float = 0.0
        self.total_upstream_cost: float = 0.0
        
        # Generate unique game ID
        self.game_id = str(uuid.uuid4())[:8]
        
        # Controllog state
        self._controllog_initialized = False
        self._run_id: Optional[str] = None
        self._task_id: Optional[str] = None
    
    def _print(self, *args, **kwargs):
        """Print to console unless in quiet mode."""
        if not self.quiet:
            console.print(*args, **kwargs)

    def init_controllog(self, log_path: Path, run_id: str) -> None:
        """Initialize controllog SDK for unified analytics."""
        try:
            cl.init(project_id="chainlex", log_dir=log_path)
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
                project_id="chainlex",
                agent_id="agent:chainlex",
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
            model_id = player.adapter.resolve_model(player.model_name)
            
            cl.model_prompt(
                task_id=self._task_id,
                agent_id=f"agent:chainlex:{call_type}",
                run_id=self._run_id,
                project_id="chainlex",
                provider="openrouter",
                model=model_id,
                prompt_tokens=prompt_tokens,
                payload={
                    "game_id": self.game_id,
                    "call_type": call_type,
                    **(payload or {}),
                },
                exchange_id=exchange_id,
            )
            
            cl.model_completion(
                task_id=self._task_id,
                agent_id=f"agent:chainlex:{call_type}",
                run_id=self._run_id,
                project_id="chainlex",
                provider="openrouter",
                model=model_id,
                completion_tokens=completion_tokens,
                wall_ms=int(latency_ms),
                cost_money=cost,
                upstream_cost_money=upstream_cost,
                payload={
                    "game_id": self.game_id,
                    "call_type": call_type,
                    **(payload or {}),
                },
                exchange_id=exchange_id,
            )
        except Exception as e:
            logger.debug(f"Failed to emit model events: {e}")

    def load_words(self) -> List[str]:
        """Load words from YAML file (cached for performance)."""
        global _WORDS_CACHE

        # Use cached words if available
        if self.words_file in _WORDS_CACHE:
            return _WORDS_CACHE[self.words_file]

        try:
            with open(self.words_file, "r") as f:
                data = yaml.safe_load(f)
                words = data.get("names", [])
                if len(words) < self.BOARD_SIZE:
                    raise ValueError(
                        f"Need at least {self.BOARD_SIZE} words, got {len(words)}"
                    )
                # Cache the words for future use
                _WORDS_CACHE[self.words_file] = words
                logger.debug(f"Loaded and cached {len(words)} words from {self.words_file}")
                return words
        except FileNotFoundError:
            logger.error(f"Words file not found: {self.words_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading words: {e}")
            raise

    def setup_board(self):
        """Initialize the game board with random word assignment."""
        # Set seed if provided for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
        
        all_words = self.load_words()

        # Select 16 random words
        self.board = random.sample(all_words, self.BOARD_SIZE)

        # Assign identities
        positions = list(range(self.BOARD_SIZE))
        random.shuffle(positions)

        # Assign: 8 friendly, 7 bystanders, 1 assassin
        friendly_positions = positions[:self.FRIENDLY_WORDS]
        bystander_positions = positions[self.FRIENDLY_WORDS:self.FRIENDLY_WORDS + self.BYSTANDERS]
        assassin_position = positions[self.FRIENDLY_WORDS + self.BYSTANDERS]

        # Create identity mapping
        self.identities = {}
        self.revealed = {}

        for i, word in enumerate(self.board):
            if i in friendly_positions:
                self.identities[word] = "friendly"
            elif i in bystander_positions:
                self.identities[word] = "bystander"
            elif i == assassin_position:
                self.identities[word] = "assassin"

            self.revealed[word] = False

        logger.info(
            f"Board setup: {self.FRIENDLY_WORDS} friendly, {self.BYSTANDERS} bystanders, {self.ASSASSINS} assassin"
        )

    def get_board_state(self, reveal_all: bool = False, revealed: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """Get current board state for display or prompts."""
        if revealed is None:
            revealed = {word: False for word in self.board}
        
        identities: Dict[str, str] = {} if not reveal_all else self.identities.copy()

        # Add revealed identities
        if not reveal_all:
            for word in self.board:
                if revealed.get(word, False):
                    identities[word] = self.identities[word]

        return {
            "board": self.board.copy(),
            "revealed": revealed.copy(),
            "identities": identities,
        }

    def display_board(self, reveal_all: bool = False):
        """Display the current board state with all identities."""
        self._print(f"\n[bold]ChainLex-1 Board[/bold]")

        # Create a 4x4 grid
        table = Table(show_header=False, show_lines=True)
        for _ in range(4):
            table.add_column(justify="center", min_width=12)

        for row in range(4):
            row_items = []
            for col in range(4):
                idx = row * 4 + col
                word = self.board[idx]

                # Color coding based on identity
                if reveal_all and word in self.identities:
                    identity = self.identities[word]
                    if identity == "friendly":
                        color = "green"
                    elif identity == "assassin":
                        color = "black on white"
                    else:  # bystander
                        color = "dim"
                else:
                    color = "white"

                row_items.append(f"[{color}]{word}[/{color}]")

            table.add_row(*row_items)

        self._print(table)

    def display_board_start(self):
        """Display the initial board state at game start (public view)."""
        self._print(f"\n[bold]Game Board[/bold]")
        
        # Create a 4x4 grid
        table = Table(show_header=False, show_lines=True)
        for _ in range(4):
            table.add_column(justify="center", min_width=12)

        for row in range(4):
            row_items = []
            for col in range(4):
                idx = row * 4 + col
                word = self.board[idx]
                row_items.append(f"[white]{word}[/white]")
            table.add_row(*row_items)

        self._print(table)
        
        # Show word type counts
        self._print(f"\n[green]Friendly Words:[/green] {self.FRIENDLY_WORDS}")
        self._print(f"[dim]Bystanders:[/dim] {self.BYSTANDERS}")
        self._print(f"[black on white]Assassin:[/black on white] {self.ASSASSINS}")
        self._print("")

    def _play_single_turn(self, player, player_label: str, head_to_head_context: str = "") -> Dict:
        """Play a single turn for one player and return their results."""
        # Fresh revealed state for this player
        revealed = {word: False for word in self.board}
        score = 0
        correct_guesses = 0
        guesses = []
        clue = None
        clue_number = None
        end_reason = None
        
        self._print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        self._print(f"[bold cyan]{player_label}'s Turn[/bold cyan]")
        self._print(f"[bold cyan]{'='*60}[/bold cyan]")
        
        # Build board state for clue giver (sees all identities)
        board_state = {
            "board": self.board.copy(),
            "revealed": revealed.copy(),
            "identities": self.identities.copy(),
            "score": score,
            "correct_guesses": correct_guesses,
        }
        
        # Clue giver phase
        clue, clue_number = player.get_clue_giver_move(
            board_state, self.prompt_files["clue_giver"], head_to_head_context
        )
        
        self._print(f'[green]Clue:[/green] "{clue}" ({clue_number})')
        
        # Log AI call metadata
        if isinstance(player, AIPlayer):
            metadata = player.get_last_call_metadata()
            if metadata:
                self.total_cost += metadata.get("openrouter_cost", 0.0)
                self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                
                self._emit_model_events(
                    player=player,
                    call_type="clue_giver",
                    prompt_tokens=metadata["input_tokens"],
                    completion_tokens=metadata["output_tokens"],
                    latency_ms=metadata["latency_ms"],
                    cost=metadata.get("openrouter_cost"),
                    upstream_cost=metadata.get("upstream_cost"),
                    payload={
                        "player": player_label,
                        "clue": clue,
                        "number": clue_number,
                    },
                )
        
        # Validate clue (simple rules, no LLM needed)
        if clue:
            _, _, is_valid, reasoning = self._validate_clue(
                clue, clue_number, board_state
            )
            if not is_valid:
                self._print(f"[red]⚠️ Invalid clue: {reasoning}[/red]")
                return {
                    "player": player_label,
                    "model": player.model_name if hasattr(player, 'model_name') else "unknown",
                    "score": 0,
                    "correct_guesses": 0,
                    "clue": clue,
                    "clue_number": clue_number,
                    "guesses": [],
                    "end_reason": "invalid_clue",
                }
        
        if clue is None or clue_number is None:
            return {
                "player": player_label,
                "model": player.model_name if hasattr(player, 'model_name') else "unknown",
                "score": 0,
                "correct_guesses": 0,
                "clue": None,
                "clue_number": None,
                "guesses": [],
                "end_reason": "no_clue",
            }
        
        # Guesser phase - build board state WITHOUT identities
        guesser_board_state = {
            "board": self.board.copy(),
            "revealed": revealed.copy(),
            "identities": {},  # Guesser doesn't see identities
            "score": score,
            "correct_guesses": correct_guesses,
        }
        
        player_guesses = player.get_guesser_moves(
            guesser_board_state, clue, clue_number, self.prompt_files["guesser"], head_to_head_context
        )
        
        # Log guesser metadata
        if isinstance(player, AIPlayer):
            metadata = player.get_last_call_metadata()
            if metadata:
                self.total_cost += metadata.get("openrouter_cost", 0.0)
                self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                
                self._emit_model_events(
                    player=player,
                    call_type="guesser",
                    prompt_tokens=metadata["input_tokens"],
                    completion_tokens=metadata["output_tokens"],
                    latency_ms=metadata["latency_ms"],
                    cost=metadata.get("openrouter_cost"),
                    upstream_cost=metadata.get("upstream_cost"),
                    payload={
                        "player": player_label,
                        "clue": clue,
                        "number": clue_number,
                        "guesses": player_guesses,
                    },
                )
        
        # Process guesses
        for guess in player_guesses:
            if guess not in self.identities:
                self._print(f"[red]Invalid guess: {guess}[/red]")
                continue
            
            identity = self.identities[guess]
            revealed[guess] = True
            
            guess_record = {
                "word": guess,
                "identity": identity,
                "correct": identity == "friendly",
                "guess_number": correct_guesses + 1 if identity == "friendly" else correct_guesses,
            }
            guesses.append(guess_record)
            
            if identity == "assassin":
                self._print(f"[black on white]💀 {guess} - THE ASSASSIN![/black on white]")
                score += self.ASSASSIN_PENALTY
                end_reason = "assassin"
                break
            elif identity == "friendly":
                correct_guesses += 1
                points = correct_guesses
                score += points
                self._print(f"[green]✓ {guess} - Correct! +{points} pts (Total: {score})[/green]")
                
                # Check if all friendly words found
                remaining = sum(
                    1 for w, i in self.identities.items()
                    if i == "friendly" and not revealed[w]
                )
                if remaining == 0:
                    self._print(f"[green]🎉 All friendly words found![/green]")
                    end_reason = "completed"
                    break
            else:  # bystander
                self._print(f"[yellow]✗ {guess} - Bystander! {self.BYSTANDER_PENALTY} pts[/yellow]")
                score += self.BYSTANDER_PENALTY
                end_reason = "bystander"
                break
        
        if end_reason is None:
            end_reason = "exhausted_guesses"
        
        self._print(f"\n[bold]{player_label} Final Score: {score}[/bold]")
        
        return {
            "player": player_label,
            "model": player.model_name if hasattr(player, 'model_name') else "unknown",
            "score": score,
            "correct_guesses": correct_guesses,
            "clue": clue,
            "clue_number": clue_number,
            "guesses": guesses,
            "end_reason": end_reason,
        }

    def get_clue_giver_turn(self) -> Tuple[Optional[str], Optional[int]]:
        """Get clue and number from the player (acting as clue giver)."""
        board_state = self.get_board_state(reveal_all=True)
        
        clue, number = self.player.get_clue_giver_move(
            board_state, self.prompt_files["clue_giver"]
        )
        
        self._print(f'[green]Clue Giver[/green]: "{clue}" ({number})')
        
        # Log AI call metadata
        if isinstance(self.player, AIPlayer):
            metadata = self.player.get_last_call_metadata()
            if metadata:
                self.total_cost += metadata.get("openrouter_cost", 0.0)
                self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                
                self._emit_model_events(
                    player=self.player,
                    call_type="clue_giver",
                    prompt_tokens=metadata["input_tokens"],
                    completion_tokens=metadata["output_tokens"],
                    latency_ms=metadata["latency_ms"],
                    cost=metadata.get("openrouter_cost"),
                    upstream_cost=metadata.get("upstream_cost"),
                    payload={
                        "clue": clue,
                        "number": number,
                    },
                )
        
        # Validate clue (simple rules, no LLM needed)
        _, _, is_valid, reasoning = self._validate_clue(
            clue, number, board_state
        )
        if not is_valid:
            self._print(f"[red]⚠️ Invalid clue: {reasoning}[/red]")
            return None, None
        
        return clue, number

    def _validate_clue(
        self, clue: str, number: int, board_state: Dict
    ) -> Tuple[str, int, bool, str]:
        """Validate clue with simple rules (no LLM needed).
        
        Rules:
        1. No multiple words (spaces) - except hyphens for compound words
        2. Clue cannot exactly match a word on the board
        
        Returns (clue, number, is_valid, reasoning).
        """
        clue_upper = clue.strip().upper()
        board_words = {word.upper() for word in board_state["board"]}
        
        # Rule 1: No multiple words (spaces not allowed)
        if ' ' in clue.strip():
            return clue, number, False, f"Multiple words not allowed: '{clue}'"
        
        # Rule 2: Clue cannot exactly match a board word
        if clue_upper in board_words:
            return clue, number, False, f"Clue matches word on board: '{clue}'"
        
        # Valid!
        return clue, number, True, "Valid clue"

    def get_guesser_guesses(self, clue: str, number: int) -> List[str]:
        """Get guesses from the player (acting as guesser)."""
        board_state = self.get_board_state(reveal_all=False)
        
        guesses = self.player.get_guesser_moves(
            board_state, clue, number, self.prompt_files["guesser"]
        )
        
        # Log AI call metadata
        if isinstance(self.player, AIPlayer):
            metadata = self.player.get_last_call_metadata()
            if metadata:
                self.total_cost += metadata.get("openrouter_cost", 0.0)
                self.total_upstream_cost += metadata.get("upstream_cost", 0.0)
                
                self._emit_model_events(
                    player=self.player,
                    call_type="guesser",
                    prompt_tokens=metadata["input_tokens"],
                    completion_tokens=metadata["output_tokens"],
                    latency_ms=metadata["latency_ms"],
                    cost=metadata.get("openrouter_cost"),
                    upstream_cost=metadata.get("upstream_cost"),
                    payload={
                        "clue": clue,
                        "number": number,
                        "guesses": guesses,
                    },
                )
        
        # Process guesses one by one
        for guess in guesses:
            self._print(f"[cyan]Guesser[/cyan] guesses: {guess}")
            result = self.process_guess(guess)
            
            if not result:  # Wrong guess ends turn
                break
        
        return guesses

    def process_guess(self, word: str) -> bool:
        """Process a single guess. Returns True if correct, False if wrong."""
        if word not in self.identities:
            logger.warning(f"Invalid guess: {word}")
            return False

        identity = self.identities[word]
        self.revealed[word] = True

        # Record the guess
        guess_record = {
            "word": word,
            "identity": identity,
            "correct": identity == "friendly",
            "guess_number": self.correct_guesses + 1 if identity == "friendly" else self.correct_guesses,
        }
        self.guesses.append(guess_record)

        if identity == "assassin":
            self._print(f"[black on white]💀 THE ASSASSIN! Game over![/black on white]")
            self.score += self.ASSASSIN_PENALTY
            self.game_over = True
            self.end_reason = "assassin"
            return False

        elif identity == "friendly":
            self.correct_guesses += 1
            points = self.correct_guesses  # Triangular scoring
            self.score += points
            self._print(f"[green]✓ Correct! +{points} points (Total: {self.score})[/green]")
            
            # Check if all friendly words found
            remaining = sum(
                1 for w, i in self.identities.items()
                if i == "friendly" and not self.revealed[w]
            )
            if remaining == 0:
                self._print(f"[green]🎉 All friendly words found![/green]")
                self.game_over = True
                self.end_reason = "completed"
            
            return True

        else:  # bystander
            self._print(f"[yellow]✗ {word} is a bystander. {self.BYSTANDER_PENALTY} points.[/yellow]")
            self.score += self.BYSTANDER_PENALTY
            self.game_over = True
            self.end_reason = "bystander"
            return False

    def calculate_max_possible_score(self) -> int:
        """Calculate maximum possible score (all 8 friendly words)."""
        # Triangular number: 1 + 2 + 3 + ... + n = n(n+1)/2
        return self.FRIENDLY_WORDS * (self.FRIENDLY_WORDS + 1) // 2

    def play(self) -> Dict:
        """Play a complete head-to-head game and return results."""
        self.start_time = time.time()

        logger.info("Starting new ChainLex-1 head-to-head game")
        self.setup_board()

        model_away = self.player_away.model_name if hasattr(self.player_away, 'model_name') else "player_away"
        model_home = self.player_home.model_name if hasattr(self.player_home, 'model_name') else "player_home"
        
        self._emit_state_move("NEW", "WIP", {
            "game_id": self.game_id,
            "model_away": model_away,
            "model_home": model_home,
        })

        self._print("[bold]🎯 ChainLex-1 Head-to-Head Game![/bold]")
        self._print(f"[cyan]Away (1st):[/cyan] {model_away}")
        self._print(f"[magenta]Home (2nd):[/magenta] {model_home}")
        self._print(f"[green]Game ID:[/green] {self.game_id}")
        self._print()
        
        # Display the board (same for both players)
        self.display_board_start()

        # Away player's turn (goes first)
        away_context = "You are going FIRST in a head-to-head game. Your opponent will see your score when they make their clue."
        result_away = self._play_single_turn(self.player_away, f"Away ({model_away})", away_context)
        
        # Home player's turn (goes second, knows opponent's score)
        home_context = f"You are going SECOND in a head-to-head game. Your opponent scored {result_away['score']} points. Make your clue with this in mind."
        result_home = self._play_single_turn(self.player_home, f"Home ({model_home})", home_context)

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Determine winner
        score_away = result_away["score"]
        score_home = result_home["score"]
        
        # Check if both players hit the assassin - instant loss for both = tie
        both_hit_assassin = (
            result_away["end_reason"] == "assassin" and 
            result_home["end_reason"] == "assassin"
        )
        
        if both_hit_assassin:
            # Both hit assassin = draw, regardless of scores
            winner = "tie"
            winner_model = None
        elif score_away > score_home:
            winner = "model_away"
            winner_model = model_away
        elif score_home > score_away:
            winner = "model_home"
            winner_model = model_home
        else:
            winner = "tie"
            winner_model = None

        # Compile results
        max_score = self.calculate_max_possible_score()
        result = {
            "game_id": self.game_id,
            "model_away": model_away,
            "model_home": model_home,
            "score_away": score_away,
            "score_home": score_home,
            "winner": winner,
            "winner_model": winner_model,
            "margin": abs(score_away - score_home),
            "max_possible_score": max_score,
            "both_hit_assassin": both_hit_assassin,
            "result_away": result_away,
            "result_home": result_home,
            "duration": duration,
            "cost": self.total_cost,
            "upstream_cost": self.total_upstream_cost,
            "board": self.board,
            "identities": self.identities,
        }

        # Display final results
        self._print(f"\n[bold]{'='*60}[/bold]")
        self._print(f"[bold]GAME RESULTS[/bold]")
        self._print(f"[bold]{'='*60}[/bold]")
        
        self._print(f"\n[cyan]Away - 1st ({model_away}):[/cyan]")
        self._print(f"  Score: {score_away} / {max_score}")
        self._print(f"  Correct: {result_away['correct_guesses']} / {self.FRIENDLY_WORDS}")
        self._print(f"  Clue: \"{result_away['clue']}\" ({result_away['clue_number']})")
        self._print(f"  End: {result_away['end_reason']}")
        
        self._print(f"\n[magenta]Home - 2nd ({model_home}):[/magenta]")
        self._print(f"  Score: {score_home} / {max_score}")
        self._print(f"  Correct: {result_home['correct_guesses']} / {self.FRIENDLY_WORDS}")
        self._print(f"  Clue: \"{result_home['clue']}\" ({result_home['clue_number']})")
        self._print(f"  End: {result_home['end_reason']}")
        
        if winner == "model_away":
            self._print(f"\n[bold]WINNER: [cyan]{model_away}[/cyan] (Away) by {result['margin']} points![/bold]")
        elif winner == "model_home":
            self._print(f"\n[bold]WINNER: [magenta]{model_home}[/magenta] (Home) by {result['margin']} points![/bold]")
        elif both_hit_assassin:
            self._print(f"\n[bold yellow]DRAW: Both players hit the assassin![/bold yellow]")
        else:
            self._print(f"\n[bold yellow]DRAW: Tied at {score_away} points![/bold yellow]")
        
        self._print(f"\nDuration: {duration:.1f}s | Cost: ${self.total_cost:.4f}")
        
        # Show board with identities
        self._print(f"\n[bold]Board (revealed):[/bold]")
        self.display_board(reveal_all=True)

        # Emit final state
        self._emit_state_move("WIP", "DONE", {
            "game_id": self.game_id,
            "winner": winner,
            "score_away": score_away,
            "score_home": score_home,
            "duration_sec": duration,
        })

        logger.info(f"Game completed. Winner: {winner}, Scores: {score_away} vs {score_home}")
        return result

