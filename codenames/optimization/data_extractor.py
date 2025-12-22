"""Extract training examples from controllog events for DSPy optimization."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum


class GuessOutcome(Enum):
    """Outcome of a single guess."""
    CORRECT = "correct"
    BYSTANDER = "bystander"
    ENEMY = "enemy"
    ASSASSIN = "assassin"


@dataclass
class GuessResult:
    """Result of a single guess."""
    word: str
    outcome: GuessOutcome
    
    @property
    def score(self) -> float:
        """Score contribution of this guess."""
        scores = {
            GuessOutcome.CORRECT: 1.0,
            GuessOutcome.BYSTANDER: -0.5,
            GuessOutcome.ENEMY: -1.0,
            GuessOutcome.ASSASSIN: -999.0,
        }
        return scores[self.outcome]


@dataclass
class ClueExample:
    """A single clue-guess training example for DSPy.
    
    Represents one spymaster clue and the resulting operative guesses.
    """
    # Game context
    game_id: str
    turn: str
    team: str
    
    # Board state (what spymaster sees)
    board_words: List[str]  # All 25 words
    team_agents: List[str]  # Words that are this team's agents
    enemy_agents: List[str]  # Words that are enemy agents
    bystanders: List[str]  # Neutral words
    assassin: str  # The assassin word
    revealed_words: List[str]  # Already revealed words
    
    # Spymaster output
    clue: str
    clue_number: int  # How many words the clue targets
    
    # Operative output
    guesses: List[str]
    guess_results: List[GuessResult] = field(default_factory=list)
    
    # Computed metrics
    @property
    def total_score(self) -> float:
        """Total score for this clue-guess sequence."""
        return sum(g.score for g in self.guess_results)
    
    @property
    def correct_count(self) -> int:
        """Number of correct guesses."""
        return sum(1 for g in self.guess_results if g.outcome == GuessOutcome.CORRECT)
    
    @property
    def hit_assassin(self) -> bool:
        """Whether the assassin was guessed."""
        return any(g.outcome == GuessOutcome.ASSASSIN for g in self.guess_results)
    
    @property
    def accuracy(self) -> float:
        """Fraction of intended words that were correctly guessed."""
        if self.clue_number == 0:
            return 1.0 if self.correct_count > 0 else 0.0
        return self.correct_count / self.clue_number
    
    @property
    def available_words(self) -> List[str]:
        """Words available for guessing (not yet revealed)."""
        return [w for w in self.board_words if w not in self.revealed_words]
    
    def to_spymaster_input(self) -> Dict[str, Any]:
        """Format as input for the spymaster module."""
        return {
            "board_words": self.available_words,
            "team_agents": [w for w in self.team_agents if w not in self.revealed_words],
            "enemy_agents": [w for w in self.enemy_agents if w not in self.revealed_words],
            "bystanders": [w for w in self.bystanders if w not in self.revealed_words],
            "assassin": self.assassin,
        }
    
    def to_operative_input(self) -> Dict[str, Any]:
        """Format as input for the operative module."""
        return {
            "available_words": self.available_words,
            "clue": self.clue,
            "clue_number": self.clue_number,
        }


def parse_game_metadata(metadata_file: Path) -> Dict[str, ClueExample]:
    """Parse a game_metadata JSONL file into ClueExample objects.
    
    Returns a dict mapping (game_id, turn, team) -> partial ClueExample
    """
    examples: Dict[str, ClueExample] = {}
    
    # First pass: collect game setup info
    game_setups: Dict[str, Dict] = {}
    
    with open(metadata_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            
            if event.get("type") == "game_setup":
                game_id = event["game_id"]
                words = event.get("words", {})
                game_setups[game_id] = {
                    "red_agents": words.get("red_agents", []),
                    "blue_agents": words.get("blue_agents", []),
                    "bystanders": words.get("bystanders", []),
                    "assassin": words.get("assassin", [""])[0] if words.get("assassin") else "",
                    "board_words": (
                        words.get("red_agents", []) + 
                        words.get("blue_agents", []) + 
                        words.get("bystanders", []) + 
                        words.get("assassin", [])
                    ),
                }
    
    # Second pass: collect clue-guess pairs
    # Track revealed words per game
    revealed_by_game: Dict[str, List[str]] = {}
    
    with open(metadata_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            
            game_id = event.get("game_id")
            if not game_id or game_id not in game_setups:
                continue
            
            setup = game_setups[game_id]
            
            if game_id not in revealed_by_game:
                revealed_by_game[game_id] = []
            
            event_type = event.get("type")
            team = event.get("team", "")
            turn = event.get("turn", "")
            
            # Process spymaster clues
            if event_type == "spymaster" and "clue" in event:
                key = f"{game_id}:{turn}:{team}"
                
                # Determine team's agents vs enemy agents
                if team == "red":
                    team_agents = setup["red_agents"]
                    enemy_agents = setup["blue_agents"]
                else:
                    team_agents = setup["blue_agents"]
                    enemy_agents = setup["red_agents"]
                
                examples[key] = ClueExample(
                    game_id=game_id,
                    turn=turn,
                    team=team,
                    board_words=setup["board_words"],
                    team_agents=team_agents,
                    enemy_agents=enemy_agents,
                    bystanders=setup["bystanders"],
                    assassin=setup["assassin"],
                    revealed_words=revealed_by_game[game_id].copy(),
                    clue=event["clue"],
                    clue_number=event.get("clue_number", 1),
                    guesses=[],
                    guess_results=[],
                )
            
            # Process operative guesses
            elif event_type == "operative" and "guesses" in event:
                key = f"{game_id}:{turn}:{team}"
                
                if key in examples:
                    example = examples[key]
                    example.guesses = event.get("guesses", [])
                    
                    # Parse guess details
                    guess_details = event.get("guess_details", [])
                    for detail in guess_details:
                        word = detail.get("guess", "")
                        result = detail.get("result", "")
                        
                        outcome_map = {
                            "correct": GuessOutcome.CORRECT,
                            "bystander": GuessOutcome.BYSTANDER,
                            "enemy": GuessOutcome.ENEMY,
                            "assassin": GuessOutcome.ASSASSIN,
                        }
                        
                        if result in outcome_map:
                            example.guess_results.append(
                                GuessResult(word=word, outcome=outcome_map[result])
                            )
                            # Track revealed words
                            if word not in revealed_by_game[game_id]:
                                revealed_by_game[game_id].append(word)
    
    return examples


def extract_training_examples(
    logs_dir: Path,
    min_clue_number: int = 1,
    exclude_assassin_hits: bool = False,
) -> List[ClueExample]:
    """Extract training examples from all game_metadata files in logs directory.
    
    Args:
        logs_dir: Path to logs directory containing game_metadata_*.jsonl files
        min_clue_number: Minimum clue number to include (filters out "0" clues)
        exclude_assassin_hits: Whether to exclude examples where assassin was hit
        
    Returns:
        List of ClueExample objects suitable for DSPy training
    """
    all_examples: List[ClueExample] = []
    
    logs_path = Path(logs_dir)
    
    # Find all game_metadata files
    metadata_files = list(logs_path.glob("game_metadata_*.jsonl"))
    
    for metadata_file in metadata_files:
        try:
            examples = parse_game_metadata(metadata_file)
            all_examples.extend(examples.values())
        except Exception as e:
            print(f"Warning: Failed to parse {metadata_file}: {e}")
            continue
    
    # Filter examples
    filtered = []
    for ex in all_examples:
        # Skip if no guesses were made
        if not ex.guess_results:
            continue
        
        # Skip low clue numbers if requested
        if ex.clue_number < min_clue_number:
            continue
        
        # Skip assassin hits if requested
        if exclude_assassin_hits and ex.hit_assassin:
            continue
        
        filtered.append(ex)
    
    return filtered


def summarize_examples(examples: List[ClueExample]) -> Dict[str, Any]:
    """Generate summary statistics for a set of training examples."""
    if not examples:
        return {"count": 0}
    
    total_score = sum(ex.total_score for ex in examples)
    avg_score = total_score / len(examples)
    
    assassin_hits = sum(1 for ex in examples if ex.hit_assassin)
    
    avg_clue_number = sum(ex.clue_number for ex in examples) / len(examples)
    avg_correct = sum(ex.correct_count for ex in examples) / len(examples)
    avg_accuracy = sum(ex.accuracy for ex in examples) / len(examples)
    
    return {
        "count": len(examples),
        "total_score": total_score,
        "avg_score": avg_score,
        "assassin_hits": assassin_hits,
        "assassin_rate": assassin_hits / len(examples),
        "avg_clue_number": avg_clue_number,
        "avg_correct_guesses": avg_correct,
        "avg_accuracy": avg_accuracy,
    }


if __name__ == "__main__":
    # Quick test
    import sys
    
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    
    examples = extract_training_examples(logs_dir)
    summary = summarize_examples(examples)
    
    print(f"\nExtracted {summary['count']} training examples")
    print(f"Average score per clue: {summary['avg_score']:.2f}")
    print(f"Assassin hit rate: {summary['assassin_rate']:.2%}")
    print(f"Average clue number: {summary['avg_clue_number']:.1f}")
    print(f"Average correct guesses: {summary['avg_correct_guesses']:.2f}")
    print(f"Average accuracy: {summary['avg_accuracy']:.1%}")
    
    # Show a few examples
    print("\n--- Sample Examples ---")
    for ex in examples[:3]:
        print(f"\nGame {ex.game_id}, Turn {ex.turn}")
        print(f"  Clue: '{ex.clue}' ({ex.clue_number})")
        print(f"  Guesses: {ex.guesses}")
        print(f"  Score: {ex.total_score:.1f}")

