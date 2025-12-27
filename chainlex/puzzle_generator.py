"""ChainLex Puzzle Generator with Semantic Clustering.

Generates puzzles algorithmically based on semantic embeddings to create
meaningful tension: friendly words should be linkable but not trivially so,
and bystanders/assassin should create plausible traps.

Design Principles:
1. Friendly words form 2-3 latent clusters (not trivial, not impossible)
2. Bystanders create false bridges between friendly clusters
3. Assassin traps the most obvious clue
4. Difficulty comes from overlap, not obscurity
"""

import json
import logging
import random
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Difficulty(Enum):
    """Puzzle difficulty tiers."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PuzzleMetrics:
    """Validation metrics for a generated puzzle."""
    friendly_cohesion: float  # Average pairwise similarity of friendly words
    friendly_bystander_avg_sim: float  # Avg similarity between friendlies and bystanders
    assassin_max_friendly_sim: float  # Max similarity between assassin and any friendly
    num_confusing_bystanders: int  # Bystanders within 0.4 sim of a friendly
    num_singleton_friendlies: int  # Friendlies not within 0.3 sim of another friendly
    dominant_cluster: List[str]  # Words in the tighter cluster
    bridge_bystanders: List[str]  # Bystanders that bridge clusters


@dataclass
class Puzzle:
    """A ChainLex puzzle with metadata."""
    puzzle_id: str
    difficulty: str
    board: List[str]
    friendly: List[str]
    bystanders: List[str]
    assassin: str
    metrics: PuzzleMetrics
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "id": self.puzzle_id,
            "difficulty": self.difficulty,
            "board": self.board,
            "friendly": self.friendly,
            "bystanders": self.bystanders,
            "assassin": self.assassin,
            "tags": self.tags,
            "metrics": asdict(self.metrics),
        }


class PuzzleGenerator:
    """Generates ChainLex puzzles using semantic embeddings."""
    
    # Difficulty tier thresholds
    DIFFICULTY_PARAMS = {
        Difficulty.EASY: {
            "friendly_cohesion_min": 0.45,
            "friendly_cohesion_max": 0.60,
            "bystander_radius": 0.35,
            "assassin_min_sim": 0.30,
            "min_confusing_bystanders": 2,
        },
        Difficulty.MEDIUM: {
            "friendly_cohesion_min": 0.35,
            "friendly_cohesion_max": 0.45,
            "bystander_radius": 0.40,
            "assassin_min_sim": 0.40,
            "min_confusing_bystanders": 4,
        },
        Difficulty.HARD: {
            "friendly_cohesion_min": 0.25,
            "friendly_cohesion_max": 0.35,
            "bystander_radius": 0.45,
            "assassin_min_sim": 0.50,
            "min_confusing_bystanders": 6,
        },
    }
    
    # Validation thresholds
    VALIDATION = {
        "friendly_cohesion_min": 0.20,
        "friendly_cohesion_max": 0.65,
        "min_confusing_bystanders": 2,
        "assassin_min_sim": 0.25,
        "friendly_min_neighbor_sim": 0.25,
    }
    
    def __init__(
        self,
        word_pool_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
    ):
        """Initialize the puzzle generator.
        
        Args:
            word_pool_path: Path to word pool YAML file
            model_name: Sentence transformer model for embeddings
            cache_embeddings: Whether to cache computed embeddings
        """
        self.word_pool_path = word_pool_path or str(
            Path(__file__).parent / "inputs" / "word_pool.yaml"
        )
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        # Lazy-loaded components
        self._model = None
        self._words: List[Dict] = []
        self._word_to_idx: Dict[str, int] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._similarity_matrix: Optional[np.ndarray] = None
        self._categories: Dict[str, List[str]] = {}
        
    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._model
    
    def _load_word_pool(self):
        """Load and index the word pool."""
        if self._words:
            return
        
        logger.info(f"Loading word pool from {self.word_pool_path}")
        with open(self.word_pool_path, "r") as f:
            data = yaml.safe_load(f)
        
        self._words = data.get("words", [])
        
        # Build word index and category mapping
        for idx, word_entry in enumerate(self._words):
            word = word_entry["word"].upper()
            self._word_to_idx[word] = idx
            
            for category in word_entry.get("categories", []):
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(word)
        
        logger.info(f"Loaded {len(self._words)} words in {len(self._categories)} categories")
    
    def _compute_embeddings(self):
        """Compute or load cached embeddings for all words."""
        if self._embeddings is not None:
            return
        
        self._load_word_pool()
        model = self._load_model()
        
        # Check for cached embeddings
        cache_path = Path(self.word_pool_path).parent / "embeddings_cache.npy"
        cache_words_path = Path(self.word_pool_path).parent / "embeddings_words.json"
        
        if self.cache_embeddings and cache_path.exists() and cache_words_path.exists():
            with open(cache_words_path, "r") as f:
                cached_words = json.load(f)
            
            # Check if cache is still valid
            current_words = [w["word"].upper() for w in self._words]
            if cached_words == current_words:
                logger.info("Loading cached embeddings")
                self._embeddings = np.load(cache_path)
                self._compute_similarity_matrix()
                return
        
        # Compute embeddings
        logger.info("Computing word embeddings...")
        words = [w["word"] for w in self._words]
        self._embeddings = model.encode(words, show_progress_bar=True)
        
        # Cache embeddings
        if self.cache_embeddings:
            np.save(cache_path, self._embeddings)
            with open(cache_words_path, "w") as f:
                json.dump([w["word"].upper() for w in self._words], f)
            logger.info(f"Cached embeddings to {cache_path}")
        
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        """Precompute pairwise similarity matrix."""
        if self._similarity_matrix is not None:
            return
        
        logger.info("Computing similarity matrix...")
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normalized = self._embeddings / norms
        self._similarity_matrix = normalized @ normalized.T
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """Get cosine similarity between two words."""
        self._compute_embeddings()
        idx1 = self._word_to_idx.get(word1.upper())
        idx2 = self._word_to_idx.get(word2.upper())
        
        if idx1 is None or idx2 is None:
            return 0.0
        
        return float(self._similarity_matrix[idx1, idx2])
    
    def get_words_by_category(self, category: str) -> List[str]:
        """Get all words in a category."""
        self._load_word_pool()
        return self._categories.get(category, [])
    
    def get_words_near(
        self,
        words: List[str],
        radius: float = 0.4,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find words within semantic radius of given words.
        
        Returns list of (word, max_similarity) tuples sorted by similarity.
        """
        self._compute_embeddings()
        exclude = exclude or set()
        
        # Get indices of input words
        indices = [self._word_to_idx.get(w.upper()) for w in words]
        indices = [i for i in indices if i is not None]
        
        if not indices:
            return []
        
        # Find max similarity to any input word for each word in pool
        candidates = []
        for word, idx in self._word_to_idx.items():
            if word in exclude:
                continue
            
            max_sim = max(self._similarity_matrix[idx, i] for i in indices)
            if max_sim >= radius * 0.5:  # Include words within half radius for selection
                candidates.append((word, float(max_sim)))
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def compute_cluster_cohesion(self, words: List[str]) -> float:
        """Compute average pairwise similarity within a word cluster."""
        self._compute_embeddings()
        
        if len(words) < 2:
            return 1.0
        
        indices = [self._word_to_idx.get(w.upper()) for w in words]
        indices = [i for i in indices if i is not None]
        
        if len(indices) < 2:
            return 0.0
        
        # Average pairwise similarity
        total_sim = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_sim += self._similarity_matrix[indices[i], indices[j]]
                count += 1
        
        return float(total_sim / count) if count > 0 else 0.0
    
    def sample_semantic_cluster(
        self,
        size: int = 4,
        seed_category: Optional[str] = None,
        exclude: Optional[Set[str]] = None,
        min_cohesion: float = 0.3,
        max_cohesion: float = 0.6,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Sample a semantic cluster of words.
        
        Args:
            size: Number of words in cluster
            seed_category: Optional category to seed from
            exclude: Words to exclude
            min_cohesion: Minimum cluster cohesion
            max_cohesion: Maximum cluster cohesion
            seed: Random seed
        
        Returns:
            List of words forming a semantic cluster
        """
        self._compute_embeddings()
        if seed is not None:
            random.seed(seed)
        
        exclude = exclude or set()
        
        # Get candidate words
        if seed_category and seed_category in self._categories:
            candidates = [w for w in self._categories[seed_category] if w not in exclude]
        else:
            candidates = [w for w in self._word_to_idx.keys() if w not in exclude]
        
        if len(candidates) < size:
            return random.sample(candidates, len(candidates))
        
        # Try to find a cluster with good cohesion
        best_cluster = None
        best_score = float('inf')  # Distance from target cohesion
        target_cohesion = (min_cohesion + max_cohesion) / 2
        
        for _ in range(50):  # Try 50 times
            # Start with a random seed word
            seed_word = random.choice(candidates)
            cluster = [seed_word]
            remaining = [w for w in candidates if w != seed_word]
            
            # Greedily add words that maintain cohesion
            while len(cluster) < size and remaining:
                # Score each candidate by how well it fits the cluster
                scored = []
                for word in remaining:
                    test_cluster = cluster + [word]
                    cohesion = self.compute_cluster_cohesion(test_cluster)
                    scored.append((word, cohesion))
                
                # Filter to candidates within cohesion bounds
                valid = [
                    (w, c) for w, c in scored
                    if min_cohesion - 0.1 <= c <= max_cohesion + 0.1
                ]
                
                if valid:
                    # Pick one that's closest to target
                    valid.sort(key=lambda x: abs(x[1] - target_cohesion))
                    chosen = valid[0][0]
                else:
                    # Fall back to any word
                    chosen = random.choice(remaining)
                
                cluster.append(chosen)
                remaining.remove(chosen)
            
            # Score this cluster
            cohesion = self.compute_cluster_cohesion(cluster)
            if min_cohesion <= cohesion <= max_cohesion:
                score = abs(cohesion - target_cohesion)
                if score < best_score:
                    best_cluster = cluster
                    best_score = score
        
        return best_cluster if best_cluster else random.sample(candidates, min(size, len(candidates)))
    
    def generate_puzzle(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: Optional[int] = None,
        max_retries: int = 20,
    ) -> Optional[Puzzle]:
        """Generate a puzzle with specified difficulty.
        
        Args:
            difficulty: Target difficulty level
            seed: Random seed for reproducibility
            max_retries: Maximum generation attempts
        
        Returns:
            Generated puzzle or None if generation failed
        """
        self._compute_embeddings()
        
        params = self.DIFFICULTY_PARAMS[difficulty]
        
        for attempt in range(max_retries):
            attempt_seed = (seed * 31 + attempt) if seed is not None else None
            if attempt_seed is not None:
                random.seed(attempt_seed)
            
            try:
                puzzle = self._generate_puzzle_attempt(difficulty, params, attempt_seed)
                if puzzle and self._validate_puzzle(puzzle, params):
                    return puzzle
            except Exception as e:
                logger.debug(f"Generation attempt {attempt + 1} failed: {e}")
        
        logger.warning(f"Failed to generate {difficulty.value} puzzle after {max_retries} attempts")
        return None
    
    def _generate_puzzle_attempt(
        self,
        difficulty: Difficulty,
        params: Dict,
        seed: Optional[int],
    ) -> Optional[Puzzle]:
        """Single attempt at puzzle generation."""
        # Select categories for clusters
        available_categories = list(self._categories.keys())
        random.shuffle(available_categories)
        
        # Step 1: Generate two friendly clusters
        cluster_1_cat = available_categories[0] if available_categories else None
        cluster_1 = self.sample_semantic_cluster(
            size=4,
            seed_category=cluster_1_cat,
            min_cohesion=params["friendly_cohesion_min"],
            max_cohesion=params["friendly_cohesion_max"],
            seed=seed,
        )
        
        if len(cluster_1) < 4:
            return None
        
        # Find a second category that's somewhat distant from first
        exclude_cats = {cluster_1_cat} if cluster_1_cat else set()
        cluster_2_cat = None
        for cat in available_categories[1:]:
            if cat not in exclude_cats:
                cluster_2_cat = cat
                break
        
        cluster_2 = self.sample_semantic_cluster(
            size=4,
            seed_category=cluster_2_cat,
            exclude=set(cluster_1),
            min_cohesion=params["friendly_cohesion_min"],
            max_cohesion=params["friendly_cohesion_max"],
            seed=seed + 1000 if seed else None,
        )
        
        if len(cluster_2) < 4:
            return None
        
        friendlies = cluster_1 + cluster_2
        
        # Step 2: Select bystanders that create confusion
        bystanders = []
        used_words = set(friendlies)
        
        # Find words near the friendly centroid
        candidates = self.get_words_near(
            friendlies,
            radius=params["bystander_radius"],
            exclude=used_words,
        )
        
        # Select bystanders, preferring confusing ones
        for word, sim in candidates:
            if len(bystanders) >= 7:
                break
            if word not in used_words:
                bystanders.append(word)
                used_words.add(word)
        
        # Fill remaining slots with random words if needed
        if len(bystanders) < 7:
            remaining = [w for w in self._word_to_idx.keys() if w not in used_words]
            needed = 7 - len(bystanders)
            bystanders.extend(random.sample(remaining, min(needed, len(remaining))))
        
        if len(bystanders) < 7:
            return None
        
        bystanders = bystanders[:7]
        used_words.update(bystanders)
        
        # Step 3: Select assassin near the dominant cluster
        cluster_1_cohesion = self.compute_cluster_cohesion(cluster_1)
        cluster_2_cohesion = self.compute_cluster_cohesion(cluster_2)
        
        dominant_cluster = cluster_1 if cluster_1_cohesion >= cluster_2_cohesion else cluster_2
        
        assassin_candidates = self.get_words_near(
            dominant_cluster,
            radius=params["assassin_min_sim"],
            exclude=used_words,
        )
        
        # Pick assassin with appropriate similarity
        assassin = None
        for word, sim in assassin_candidates:
            if sim >= params["assassin_min_sim"]:
                assassin = word
                break
        
        if not assassin and assassin_candidates:
            assassin = assassin_candidates[0][0]
        
        if not assassin:
            # Fall back to random word
            remaining = [w for w in self._word_to_idx.keys() if w not in used_words]
            assassin = random.choice(remaining) if remaining else None
        
        if not assassin:
            return None
        
        # Build the board (shuffle the 16 words)
        board = friendlies + bystanders + [assassin]
        random.shuffle(board)
        
        # Compute metrics
        metrics = self._compute_metrics(friendlies, bystanders, assassin, dominant_cluster)
        
        # Determine tags
        tags = self._determine_tags(
            friendlies, bystanders, assassin, 
            cluster_1_cat, cluster_2_cat, metrics
        )
        
        return Puzzle(
            puzzle_id=f"{difficulty.value}_{uuid.uuid4().hex[:8]}",
            difficulty=difficulty.value,
            board=board,
            friendly=friendlies,
            bystanders=bystanders,
            assassin=assassin,
            metrics=metrics,
            tags=tags,
        )
    
    def _compute_metrics(
        self,
        friendlies: List[str],
        bystanders: List[str],
        assassin: str,
        dominant_cluster: List[str],
    ) -> PuzzleMetrics:
        """Compute validation metrics for a puzzle."""
        # Friendly cohesion
        friendly_cohesion = self.compute_cluster_cohesion(friendlies)
        
        # Friendly-bystander similarity
        total_sim = 0.0
        count = 0
        confusing_bystanders = []
        for bystander in bystanders:
            max_sim = max(self.get_similarity(bystander, f) for f in friendlies)
            total_sim += max_sim
            count += 1
            if max_sim >= 0.35:
                confusing_bystanders.append(bystander)
        
        friendly_bystander_avg = total_sim / count if count > 0 else 0.0
        
        # Assassin proximity
        assassin_max_sim = max(self.get_similarity(assassin, f) for f in friendlies)
        
        # Singleton friendlies
        singletons = 0
        for f in friendlies:
            max_neighbor_sim = max(
                self.get_similarity(f, other) 
                for other in friendlies if other != f
            )
            if max_neighbor_sim < self.VALIDATION["friendly_min_neighbor_sim"]:
                singletons += 1
        
        # Bridge bystanders (connect both clusters)
        cluster_1 = friendlies[:4]
        cluster_2 = friendlies[4:]
        bridge_bystanders = []
        for b in bystanders:
            sim_to_c1 = max(self.get_similarity(b, w) for w in cluster_1)
            sim_to_c2 = max(self.get_similarity(b, w) for w in cluster_2)
            if sim_to_c1 >= 0.3 and sim_to_c2 >= 0.3:
                bridge_bystanders.append(b)
        
        return PuzzleMetrics(
            friendly_cohesion=friendly_cohesion,
            friendly_bystander_avg_sim=friendly_bystander_avg,
            assassin_max_friendly_sim=assassin_max_sim,
            num_confusing_bystanders=len(confusing_bystanders),
            num_singleton_friendlies=singletons,
            dominant_cluster=dominant_cluster,
            bridge_bystanders=bridge_bystanders,
        )
    
    def _validate_puzzle(self, puzzle: Puzzle, params: Dict) -> bool:
        """Validate puzzle meets quality criteria."""
        m = puzzle.metrics
        v = self.VALIDATION
        
        # Check cohesion bounds
        if not (v["friendly_cohesion_min"] <= m.friendly_cohesion <= v["friendly_cohesion_max"]):
            logger.debug(f"Cohesion out of bounds: {m.friendly_cohesion}")
            return False
        
        # Check confusing bystanders
        if m.num_confusing_bystanders < params.get("min_confusing_bystanders", v["min_confusing_bystanders"]):
            logger.debug(f"Not enough confusing bystanders: {m.num_confusing_bystanders}")
            return False
        
        # Check assassin proximity (should be dangerous)
        if m.assassin_max_friendly_sim < v["assassin_min_sim"]:
            logger.debug(f"Assassin not dangerous enough: {m.assassin_max_friendly_sim}")
            return False
        
        # Check for singleton friendlies (every word should be linkable)
        if m.num_singleton_friendlies > 1:  # Allow at most 1 outlier
            logger.debug(f"Too many singleton friendlies: {m.num_singleton_friendlies}")
            return False
        
        return True
    
    def _determine_tags(
        self,
        friendlies: List[str],
        bystanders: List[str],
        assassin: str,
        cat1: Optional[str],
        cat2: Optional[str],
        metrics: PuzzleMetrics,
    ) -> List[str]:
        """Determine descriptive tags for the puzzle."""
        tags = []
        
        # Category-based tags
        if cat1:
            tags.append(f"cluster:{cat1}")
        if cat2:
            tags.append(f"cluster:{cat2}")
        
        # Trap type tags
        if metrics.assassin_max_friendly_sim >= 0.5:
            tags.append("semantic_trap")
        
        if len(metrics.bridge_bystanders) >= 2:
            tags.append("bridge_hazard")
        
        # Difficulty indicators
        if metrics.friendly_cohesion >= 0.5:
            tags.append("high_cohesion")
        elif metrics.friendly_cohesion <= 0.3:
            tags.append("low_cohesion")
        
        return tags
    
    def generate_puzzle_pool(
        self,
        num_puzzles: int = 50,
        difficulty_distribution: Optional[Dict[Difficulty, float]] = None,
        seed: int = 42,
    ) -> List[Puzzle]:
        """Generate a pool of puzzles with difficulty distribution.
        
        Args:
            num_puzzles: Total number of puzzles to generate
            difficulty_distribution: Dict of difficulty -> fraction (default: even split)
            seed: Base random seed
        
        Returns:
            List of generated puzzles
        """
        if difficulty_distribution is None:
            difficulty_distribution = {
                Difficulty.EASY: 0.33,
                Difficulty.MEDIUM: 0.34,
                Difficulty.HARD: 0.33,
            }
        
        puzzles = []
        puzzle_counts = {
            d: int(num_puzzles * frac) 
            for d, frac in difficulty_distribution.items()
        }
        
        # Ensure we hit exact count
        total = sum(puzzle_counts.values())
        if total < num_puzzles:
            puzzle_counts[Difficulty.MEDIUM] += num_puzzles - total
        
        logger.info(f"Generating {num_puzzles} puzzles: {puzzle_counts}")
        
        for difficulty, count in puzzle_counts.items():
            for i in range(count):
                puzzle_seed = seed + len(puzzles) * 1000
                puzzle = self.generate_puzzle(difficulty, seed=puzzle_seed)
                if puzzle:
                    puzzles.append(puzzle)
                    logger.info(
                        f"Generated {difficulty.value} puzzle {len(puzzles)}/{num_puzzles}: "
                        f"cohesion={puzzle.metrics.friendly_cohesion:.2f}, "
                        f"assassin_sim={puzzle.metrics.assassin_max_friendly_sim:.2f}"
                    )
                else:
                    logger.warning(f"Failed to generate {difficulty.value} puzzle {i + 1}")
        
        return puzzles
    
    def save_puzzle_pool(
        self,
        puzzles: List[Puzzle],
        output_path: str,
        pool_name: str = "puzzles",
    ):
        """Save puzzle pool to YAML file."""
        output = {
            "pool_name": pool_name,
            "count": len(puzzles),
            "difficulty_distribution": {
                d.value: sum(1 for p in puzzles if p.difficulty == d.value)
                for d in Difficulty
            },
            "puzzles": [p.to_dict() for p in puzzles],
        }
        
        with open(output_path, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved {len(puzzles)} puzzles to {output_path}")


def main():
    """CLI for puzzle generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ChainLex puzzles")
    parser.add_argument("--num-training", type=int, default=50, help="Number of training puzzles")
    parser.add_argument("--num-eval", type=int, default=50, help="Number of eval puzzles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="chainlex/inputs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    generator = PuzzleGenerator()
    
    # Generate training puzzles
    print(f"Generating {args.num_training} training puzzles...")
    training_puzzles = generator.generate_puzzle_pool(
        num_puzzles=args.num_training,
        seed=args.seed,
    )
    
    training_path = Path(args.output_dir) / "puzzles_training.yaml"
    generator.save_puzzle_pool(training_puzzles, str(training_path), "training")
    print(f"Saved training puzzles to {training_path}")
    
    # Generate eval puzzles (different seed for independence)
    print(f"Generating {args.num_eval} eval puzzles...")
    eval_puzzles = generator.generate_puzzle_pool(
        num_puzzles=args.num_eval,
        seed=args.seed + 100000,  # Completely different seed space
    )
    
    eval_path = Path(args.output_dir) / "puzzles_eval.yaml"
    generator.save_puzzle_pool(eval_puzzles, str(eval_path), "eval")
    print(f"Saved eval puzzles to {eval_path}")
    
    print(f"\nGenerated {len(training_puzzles)} training + {len(eval_puzzles)} eval puzzles")


if __name__ == "__main__":
    main()

