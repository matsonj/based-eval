"""Puzzle loading and board generation for ChainLex-1.

Supports two modes:
- Pool selection: Load pre-defined puzzles from YAML files
- Legacy random: Generate random boards from word list (for backwards compat)

Usage:
    # For optimization (training pool only)
    loader = PuzzleLoader(pool=PuzzlePool.TRAINING)
    puzzle = loader.get_puzzle(seed=42)
    
    # For eval/run (eval pool only)
    loader = PuzzleLoader(pool=PuzzlePool.EVAL)
    puzzle = loader.get_puzzle(puzzle_id="eval_abc123")
    
    # Legacy random generation
    loader = PuzzleLoader(pool=None)  # or use_legacy=True
    puzzle = loader.generate_random_board(seed=42)
"""

import logging
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class PuzzlePool(Enum):
    """Available puzzle pools."""
    TRAINING = "training"  # Used by optimizer only
    EVAL = "eval"          # Used by eval and run commands


class PuzzleLoader:
    """Load and provide puzzles for ChainLex-1 games."""
    
    def __init__(
        self,
        pool: Optional[PuzzlePool] = None,
        use_legacy: bool = False,
        puzzles_dir: Optional[str] = None,
        words_file: Optional[str] = None,
    ):
        """Initialize the puzzle loader.
        
        Args:
            pool: Which puzzle pool to use (TRAINING or EVAL)
            use_legacy: If True, use legacy random generation instead of pools
            puzzles_dir: Directory containing puzzle pool YAML files
            words_file: Path to words YAML file (for legacy mode)
        """
        self.pool = pool
        self.use_legacy = use_legacy or pool is None
        
        # Set up paths
        base_dir = Path(__file__).parent
        self.puzzles_dir = Path(puzzles_dir) if puzzles_dir else base_dir / "inputs"
        self.words_file = words_file or str(base_dir.parent / "inputs" / "names.yaml")
        
        # Cached data
        self._puzzles: Optional[List[Dict]] = None
        self._puzzle_index: Dict[str, Dict] = {}
        self._words: Optional[List[str]] = None
    
    def _load_pool(self) -> List[Dict]:
        """Load the puzzle pool from YAML."""
        if self._puzzles is not None:
            return self._puzzles
        
        if self.use_legacy or self.pool is None:
            self._puzzles = []
            return self._puzzles
        
        pool_file = self.puzzles_dir / f"puzzles_{self.pool.value}.yaml"
        
        if not pool_file.exists():
            logger.warning(f"Puzzle pool file not found: {pool_file}")
            logger.warning("Falling back to legacy random generation")
            self.use_legacy = True
            self._puzzles = []
            return self._puzzles
        
        logger.info(f"Loading puzzle pool from {pool_file}")
        with open(pool_file, "r") as f:
            data = yaml.safe_load(f)
        
        self._puzzles = data.get("puzzles", [])
        
        # Build index by puzzle ID
        for puzzle in self._puzzles:
            puzzle_id = puzzle.get("id")
            if puzzle_id:
                self._puzzle_index[puzzle_id] = puzzle
        
        logger.info(f"Loaded {len(self._puzzles)} puzzles from {self.pool.value} pool")
        return self._puzzles
    
    def _load_words(self) -> List[str]:
        """Load words for legacy random generation."""
        if self._words is not None:
            return self._words
        
        with open(self.words_file, "r") as f:
            data = yaml.safe_load(f)
        
        self._words = data.get("names", [])
        logger.info(f"Loaded {len(self._words)} words for legacy generation")
        return self._words
    
    def get_puzzle(
        self,
        puzzle_id: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a puzzle from the pool.
        
        Args:
            puzzle_id: Specific puzzle ID to retrieve
            seed: Random seed for selecting a puzzle (if no ID specified)
            difficulty: Filter by difficulty (easy/medium/hard)
        
        Returns:
            Dict with board, friendly_words, bystanders, assassin, and metadata
        """
        if self.use_legacy:
            return self.generate_random_board(seed=seed)
        
        puzzles = self._load_pool()
        
        if not puzzles:
            logger.warning("No puzzles in pool, falling back to random generation")
            return self.generate_random_board(seed=seed)
        
        # Get specific puzzle by ID
        if puzzle_id and puzzle_id in self._puzzle_index:
            puzzle = self._puzzle_index[puzzle_id]
            return self._puzzle_to_board(puzzle)
        
        # Filter by difficulty if specified
        candidates = puzzles
        if difficulty:
            candidates = [p for p in puzzles if p.get("difficulty") == difficulty]
            if not candidates:
                logger.warning(f"No puzzles with difficulty={difficulty}, using all")
                candidates = puzzles
        
        # Select puzzle based on seed
        if seed is not None:
            random.seed(seed)
        
        puzzle = random.choice(candidates)
        return self._puzzle_to_board(puzzle)
    
    def get_puzzles(
        self,
        count: int,
        seed: Optional[int] = None,
        difficulty: Optional[str] = None,
        shuffle: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get multiple puzzles from the pool.
        
        Args:
            count: Number of puzzles to return
            seed: Random seed for selection
            difficulty: Filter by difficulty
            shuffle: Whether to shuffle selection
        
        Returns:
            List of puzzle dicts
        """
        if self.use_legacy:
            return [
                self.generate_random_board(seed=seed + i if seed else None)
                for i in range(count)
            ]
        
        puzzles = self._load_pool()
        
        if not puzzles:
            logger.warning("No puzzles in pool, falling back to random generation")
            return [
                self.generate_random_board(seed=seed + i if seed else None)
                for i in range(count)
            ]
        
        # Filter by difficulty
        candidates = puzzles
        if difficulty:
            candidates = [p for p in puzzles if p.get("difficulty") == difficulty]
            if not candidates:
                candidates = puzzles
        
        # Select puzzles
        if seed is not None:
            random.seed(seed)
        
        if shuffle:
            candidates = list(candidates)
            random.shuffle(candidates)
        
        # Return up to count puzzles (cycle if needed)
        result = []
        for i in range(count):
            puzzle = candidates[i % len(candidates)]
            result.append(self._puzzle_to_board(puzzle))
        
        return result
    
    def get_all_puzzles(self) -> List[Dict[str, Any]]:
        """Get all puzzles in the pool."""
        if self.use_legacy:
            raise ValueError("Cannot get all puzzles in legacy mode")
        
        puzzles = self._load_pool()
        return [self._puzzle_to_board(p) for p in puzzles]
    
    def get_puzzle_count(self) -> int:
        """Get number of puzzles in the pool."""
        if self.use_legacy:
            return 0
        return len(self._load_pool())
    
    def _puzzle_to_board(self, puzzle: Dict) -> Dict[str, Any]:
        """Convert puzzle dict to board format expected by game."""
        return {
            "puzzle_id": puzzle.get("id"),
            "board": puzzle.get("board", []),
            "friendly_words": puzzle.get("friendly", []),
            "bystanders": puzzle.get("bystanders", []),
            "assassin": puzzle.get("assassin"),
            "difficulty": puzzle.get("difficulty"),
            "tags": puzzle.get("tags", []),
            "metrics": puzzle.get("metrics", {}),
        }
    
    def generate_random_board(
        self,
        seed: Optional[int] = None,
        words_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Legacy: Generate a random board from word list.
        
        This is the original random generation for backwards compatibility.
        
        Args:
            seed: Random seed for reproducibility
            words_file: Override words file path
        
        Returns:
            Dict with board, friendly_words, bystanders, assassin
        """
        if words_file:
            with open(words_file, "r") as f:
                data = yaml.safe_load(f)
            words = data.get("names", [])
        else:
            words = self._load_words()
        
        if seed is not None:
            random.seed(seed)
        
        # Select 16 random words
        board = random.sample(words, 16)
        
        # Assign identities: 8 friendly, 7 bystanders, 1 assassin
        positions = list(range(16))
        random.shuffle(positions)
        
        friendly_positions = set(positions[:8])
        bystander_positions = set(positions[8:15])
        assassin_position = positions[15]
        
        friendly_words = [board[i] for i in friendly_positions]
        bystanders = [board[i] for i in bystander_positions]
        assassin = board[assassin_position]
        
        return {
            "puzzle_id": None,  # No ID for random boards
            "board": board,
            "friendly_words": friendly_words,
            "bystanders": bystanders,
            "assassin": assassin,
            "difficulty": None,
            "tags": ["random"],
            "metrics": {},
        }


# Convenience functions for common use cases

def get_training_loader() -> PuzzleLoader:
    """Get a puzzle loader for training (optimizer use only)."""
    return PuzzleLoader(pool=PuzzlePool.TRAINING)


def get_eval_loader() -> PuzzleLoader:
    """Get a puzzle loader for evaluation and runs."""
    return PuzzleLoader(pool=PuzzlePool.EVAL)


def get_legacy_loader(words_file: Optional[str] = None) -> PuzzleLoader:
    """Get a puzzle loader using legacy random generation."""
    return PuzzleLoader(use_legacy=True, words_file=words_file)

