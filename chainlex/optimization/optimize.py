"""Main optimization logic for ChainLex-1 using DSPy.

Uses DSPy optimizers to find better prompt configurations by:
1. Generating training examples (board states)
2. Running the pipeline and scoring results
3. Optimizing few-shot examples and instructions
"""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
import yaml

from chainlex.optimization.modules import ChainLexPipeline
from chainlex.optimization.metrics import chainlex_metric, normalized_metric, score_guesses, BYSTANDER_PENALTY, ASSASSIN_PENALTY

logger = logging.getLogger(__name__)


def gepa_feedback_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric that provides textual feedback for evolution.

    GEPA uses this feedback to understand WHY a score was low and evolve
    better instructions.

    Uses the same normalized scoring as the rest of the system for consistency.
    """
    # Use the normalized metric for consistent scoring
    score = normalized_metric(example, prediction, trace)

    # If GEPA is asking for predictor-level feedback, return dict
    if pred_name is not None:
        # Parse board state for detailed feedback
        friendly_words = set(
            word.strip().upper()
            for word in str(example.friendly_words).split(',')
        )
        bystanders = set(
            word.strip().upper()
            for word in str(example.bystanders).split(',')
        )
        assassin = str(example.assassin).strip().upper()

        # Get guesses from prediction
        guesses = prediction.guesses if hasattr(prediction, 'guesses') else []
        if isinstance(guesses, str):
            guesses = [g.strip().upper() for g in guesses.split(',')]
        else:
            guesses = [g.upper() for g in guesses]

        # Get clue info
        clue = getattr(prediction, 'clue', 'UNKNOWN')
        clue_number = getattr(prediction, 'clue_number', 0)

        # Analyze what happened
        feedback_parts = []
        hit_assassin = any(guess == assassin for guess in guesses)
        hit_bystander = any(guess in bystanders for guess in guesses)
        correct_count = sum(1 for guess in guesses if guess in friendly_words)

        # Generate detailed feedback
        if hit_assassin:
            assassin_pos = next(i for i, guess in enumerate(guesses) if guess == assassin)
            feedback_parts.append(f"CRITICAL FAILURE: Assassin '{assassin}' was guessed at position {assassin_pos + 1}. ")
            feedback_parts.append(f"The clue '{clue}' dangerously associated with the assassin word. ")
            if pred_name == "clue_giver":
                feedback_parts.append("CLUE GIVER: You must explicitly verify your clue cannot lead to the assassin. Consider safer, more specific connections.")
            else:  # guesser
                feedback_parts.append("GUESSER: You guessed too aggressively. When uncertain, stop before risking the assassin.")

        elif hit_bystander:
            bystander_guess = next(guess for guess in guesses if guess in bystanders)
            bystander_pos = next(i for i, guess in enumerate(guesses) if guess == bystander_guess)
            feedback_parts.append(f"Bystander hit: '{bystander_guess}' at position {bystander_pos + 1} ended the turn prematurely. ")
            if correct_count > 0:
                feedback_parts.append(f"You got {correct_count} correct guesses first, but the bystander penalty canceled the gains. ")
            feedback_parts.append(f"The clue '{clue}' had unintended associations with neutral words.")

        elif correct_count == 0:
            feedback_parts.append(f"FAILURE: No correct guesses. The clue '{clue}' ({clue_number}) failed to connect with any friendly words. ")
            feedback_parts.append("Consider more direct or obvious associations between friendly words.")

        elif correct_count < clue_number:
            feedback_parts.append(f"UNDERPERFORMANCE: Clue '{clue}' promised {clue_number} words but only {correct_count} were guessed correctly. ")
            feedback_parts.append("Your associations were too subtle or the guesser couldn't make the connection.")

        else:  # Success case
            if correct_count >= clue_number:
                feedback_parts.append(f"SUCCESS: Clue '{clue}' ({clue_number}) successfully connected to {correct_count} friendly words! ")
                if correct_count > clue_number:
                    feedback_parts.append(f"Bonus: Guesser found {correct_count - clue_number} extra words.")

        # Add strategic advice based on score ranges
        if score < 0.3:
            feedback_parts.append("Overall: This approach needs significant improvement. Focus on assassin avoidance above all else.")
        elif score < 0.6:
            feedback_parts.append("Overall: Making progress but still hitting penalties. Work on cleaner clue/association quality.")
        else:
            feedback_parts.append("Overall: Good performance! Continue refining for even better results.")

        feedback = " ".join(feedback_parts)
        return {"score": score, "feedback": feedback}

    # Standard evaluation - just return the score
    return score


def _extract_score(result) -> float:
    """Extract numeric score from DSPy evaluation result.
    
    DSPy's Evaluate returns an EvaluationResult where:
    - result.score is the PERCENTAGE (actual_avg * 100)
    - To get the actual average, divide by 100
    
    For ChainLex-1 with scores like -66.53 avg, DSPy shows:
    "Average Metric: -998.00 / 15 (-6653.3%)" 
    and result.score = -6653.3 (the percentage)
    """
    if isinstance(result, (int, float)):
        return float(result)
    
    # DSPy EvaluationResult: score is percentage, divide by 100 for actual average
    if hasattr(result, 'score'):
        # result.score is percentage (avg * 100), so divide by 100
        return float(result.score) / 100.0
    
    # Fallbacks for other formats
    if hasattr(result, 'average'):
        return float(result.average)
    if hasattr(result, 'mean'):
        return float(result.mean)
    
    # If it's a tuple, first element is usually the score
    if isinstance(result, tuple) and len(result) > 0:
        return float(result[0])
    
    # Last resort: try to convert directly
    try:
        return float(result)
    except (TypeError, ValueError):
        logger.warning(f"Could not extract score from {type(result)}: {result}")
        return 0.0


def load_words(words_file: str = "inputs/names.yaml") -> List[str]:
    """Load words from the YAML file."""
    with open(words_file, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def generate_board(words: List[str], seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a random ChainLex-1 board (legacy mode).
    
    Returns:
        Dict with board, friendly_words, bystanders, assassin
    """
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
        "board": board,
        "friendly_words": friendly_words,
        "bystanders": bystanders,
        "assassin": assassin,
    }


def create_training_examples(
    num_examples: int = 50,
    words_file: str = "inputs/names.yaml",
    base_seed: int = 42,
    use_puzzle_pool: bool = True,
) -> List[dspy.Example]:
    """Generate training examples for DSPy optimization.

    Uses the TRAINING puzzle pool by default for semantically-designed puzzles.
    Falls back to legacy random generation if pool is not available.
    
    IMPORTANT: Training pool is ONLY used by the optimizer, never by eval or run.
    """
    examples = []
    
    if use_puzzle_pool:
        try:
            from chainlex.puzzle_loader import PuzzleLoader, PuzzlePool
            
            loader = PuzzleLoader(pool=PuzzlePool.TRAINING)
            pool_size = loader.get_puzzle_count()
            
            if pool_size > 0:
                logger.info(f"Loading {num_examples} examples from TRAINING pool ({pool_size} available)")
                puzzles = loader.get_puzzles(count=num_examples, seed=base_seed, shuffle=True)
                
                for puzzle in puzzles:
                    example = dspy.Example(
                        board=", ".join(puzzle["board"]),
                        friendly_words=", ".join(puzzle["friendly_words"]),
                        bystanders=", ".join(puzzle["bystanders"]),
                        assassin=puzzle["assassin"],
                    ).with_inputs("board", "friendly_words", "bystanders", "assassin")
                    examples.append(example)
                
                logger.info(f"Loaded {len(examples)} training examples from puzzle pool")
                return examples
            else:
                logger.warning("Training puzzle pool is empty, falling back to legacy generation")
        except Exception as e:
            logger.warning(f"Could not load training pool: {e}, falling back to legacy generation")
    
    # Legacy random generation fallback
    logger.info(f"Using legacy random generation for {num_examples} training examples")
    words = load_words(words_file)

    for i in range(num_examples):
        # Use different seeds to ensure diversity
        seed = base_seed + i * 31  # Use larger multiplier for more diversity
        board_data = generate_board(words, seed=seed)

        example = dspy.Example(
            board=", ".join(board_data["board"]),
            friendly_words=", ".join(board_data["friendly_words"]),
            bystanders=", ".join(board_data["bystanders"]),
            assassin=board_data["assassin"],
        ).with_inputs("board", "friendly_words", "bystanders", "assassin")

        examples.append(example)

    return examples


def create_eval_examples(
    num_examples: int = 20,
    words_file: str = "inputs/names.yaml",
    base_seed: int = 42,
    use_puzzle_pool: bool = True,
) -> List[dspy.Example]:
    """Generate evaluation examples for DSPy optimization.

    Uses a SEPARATE seed space from training to ensure independence.
    For optimizer internal eval, we use training pool with different seed.
    """
    # For optimizer's internal eval, use training pool but different seed
    # This keeps train/eval independent while using designed puzzles
    return create_training_examples(
        num_examples=num_examples,
        words_file=words_file,
        base_seed=base_seed + 50000,  # Different seed space
        use_puzzle_pool=use_puzzle_pool,
    )


def run_optimization(
    output_dir: Path,
    model: str = "openai/gpt-4o-mini",
    num_train_examples: int = 50,  # Increased default for better optimization
    num_eval_examples: int = 20,   # Increased for more stable evaluation
    num_threads: int = 4,
    max_bootstrapped_demos: int = 8,  # Increased for more examples
    max_labeled_demos: int = 8,      # Increased for more examples
    seed: int = 42,
    words_file: str = "inputs/names.yaml",
    clue_giver_prompt: Optional[str] = None,
    guesser_prompt: Optional[str] = None,
    dry_run: bool = False,
    optimizer_budget: str = "medium",  # New parameter for optimizer intensity
    blend_models: Optional[List[str]] = None,  # List of models for round-robin blending
) -> Dict[str, Any]:
    """Run DSPy optimization on ChainLex-1 pipeline.
    
    Args:
        output_dir: Directory to save optimized pipeline
        model: OpenRouter model ID to use
        num_train_examples: Number of training boards
        num_eval_examples: Number of evaluation boards
        num_threads: Parallel evaluation threads
        max_bootstrapped_demos: Max few-shot examples from bootstrapping
        max_labeled_demos: Max labeled examples to include
        seed: Random seed for reproducibility
        words_file: Path to words YAML file
        clue_giver_prompt: Optional custom clue giver prompt file
        guesser_prompt: Optional custom guesser prompt file
        dry_run: If True, just test data loading without optimization
    
    Returns:
        Dict with optimization results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting ChainLex-1 optimization")
    logger.info(f"Model: {model}")
    logger.info(f"Training examples: {num_train_examples}")
    logger.info(f"Eval examples: {num_eval_examples}")
    
    # Configure DSPy with OpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Setup model(s) for optimization
    if blend_models:
        # Blend mode: create LMs for all models and set up rotation
        logger.info(f"BLEND MODE: Round-robin across {len(blend_models)} models")
        blend_lms = []
        
        # Reasoning models (like gpt-5) require temperature=1.0 and higher max_tokens
        reasoning_model_patterns = ["gpt-5", "o1", "o3"]
        
        for blend_model in blend_models:
            is_reasoning = any(p in blend_model.lower() for p in reasoning_model_patterns)
            
            if is_reasoning:
                lm = dspy.LM(
                    model=f"openrouter/{blend_model}",
                    api_key=api_key,
                    api_base="https://openrouter.ai/api/v1",
                    cache=True,
                    temperature=1.0,  # Required for reasoning models
                    max_tokens=16000,  # Required for reasoning models
                )
                logger.info(f"  - Initialized (reasoning): {blend_model}")
            else:
                lm = dspy.LM(
                    model=f"openrouter/{blend_model}",
                    api_key=api_key,
                    api_base="https://openrouter.ai/api/v1",
                    cache=True,
                    temperature=0.0,
                    max_tokens=8192,
                )
                logger.info(f"  - Initialized: {blend_model}")
            
            blend_lms.append(lm)
        
        # Create a rotating model selector (thread-safe)
        import threading
        
        class ModelRotator:
            def __init__(self, lms: List[dspy.LM]):
                self.lms = lms
                self.index = 0
                self._lock = threading.Lock()
            
            def next(self) -> dspy.LM:
                with self._lock:
                    lm = self.lms[self.index]
                    self.index = (self.index + 1) % len(self.lms)
                    return lm
            
            def current_model_name(self) -> str:
                with self._lock:
                    return self.lms[self.index].model
        
        model_rotator = ModelRotator(blend_lms)
        
        # Start with first model
        dspy.configure(lm=blend_lms[0])
    else:
        # Single model mode
        model_rotator = None
        lm = dspy.LM(
            model=f"openrouter/{model}",
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=True,  # Enable caching for efficiency during optimization
            temperature=0.0,  # Deterministic generation for reproducibility
            max_tokens=8192,  # Large enough for GEPA's evolved prompts
        )
        dspy.configure(lm=lm)
    
    # Generate training and evaluation examples from TRAINING pool
    # NOTE: Optimizer uses TRAINING pool only - never sees EVAL pool puzzles
    logger.info("Loading training examples from TRAINING puzzle pool...")
    train_examples = create_training_examples(
        num_examples=num_train_examples,
        words_file=words_file,
        base_seed=seed,
        use_puzzle_pool=True,
    )
    
    logger.info("Loading evaluation examples from TRAINING puzzle pool (different seed)...")
    eval_examples = create_eval_examples(
        num_examples=num_eval_examples,
        words_file=words_file,
        base_seed=seed,  # create_eval_examples adds offset internally
        use_puzzle_pool=True,
    )
    
    if dry_run:
        logger.info("Dry run - testing pipeline on one example...")
        pipeline = ChainLexPipeline(
            clue_giver_prompt=clue_giver_prompt,
            guesser_prompt=guesser_prompt,
        )
        
        # Test on first example
        example = train_examples[0]
        result = pipeline(
            board=example.board,
            friendly_words=example.friendly_words,
            bystanders=example.bystanders,
            assassin=example.assassin,
        )
        
        score = chainlex_metric(example, result)
        
        return {
            "dry_run": True,
            "test_clue": result.clue,
            "test_number": result.clue_number,
            "test_guesses": result.guesses,
            "test_score": score,
            "num_train_examples": len(train_examples),
            "num_eval_examples": len(eval_examples),
        }
    
    # Create pipeline
    logger.info("Creating pipeline...")
    pipeline = ChainLexPipeline(
        clue_giver_prompt=clue_giver_prompt,
        guesser_prompt=guesser_prompt,
    )
    
    # Evaluate baseline with normalized metric (consistent with optimization)
    logger.info("Evaluating baseline performance...")
    baseline_evaluator = dspy.Evaluate(
        devset=eval_examples,
        metric=normalized_metric,  # Use same metric as optimization
        num_threads=num_threads,
        display_progress=True,
    )
    baseline_result = baseline_evaluator(pipeline)
    baseline_score = _extract_score(baseline_result)
    logger.info(".3f")

    # Also compute raw score for human readability
    raw_baseline_evaluator = dspy.Evaluate(
        devset=eval_examples,
        metric=chainlex_metric,
        num_threads=num_threads,
        display_progress=False,  # Don't show progress again
    )
    raw_baseline_result = raw_baseline_evaluator(pipeline)
    raw_baseline_score = _extract_score(raw_baseline_result)
    logger.info(".1f")
    
    # Try GEPA (Genetic Evolution of Prompts Algorithm) for instruction optimization
    # GEPA uses reflection and textual feedback to evolve better prompts
    logger.info("Running GEPA optimization (evolutionary prompt optimization with feedback)...")

    try:
        from dspy.teleprompt import GEPA

        # Create reflection LM (use same model but with deterministic settings for consistency)
        reflection_lm = dspy.LM(
            model=f"openrouter/{model}",
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,  # Deterministic for reflection
            cache=False,
            max_tokens=8192,  # Large enough for GEPA's evolved prompts
        )

        # Create blending metric if in blend mode
        if model_rotator is not None:
            logger.info("Using BLEND metric: rotating models on each evaluation")
            
            def blend_gepa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
                """GEPA metric that rotates through blend models.
                
                Uses dspy.context() for thread-safe model switching.
                """
                # Rotate to next model (thread-safe)
                next_lm = model_rotator.next()
                # Use context manager for thread-safe model switching
                with dspy.context(lm=next_lm):
                    return gepa_feedback_metric(example, prediction, trace, pred_name, pred_trace)
            
            optimization_metric = blend_gepa_metric
        else:
            optimization_metric = gepa_feedback_metric

        optimizer = GEPA(
            metric=optimization_metric,  # Use feedback-aware metric (optionally with blending)
            auto=optimizer_budget,  # Configurable optimization budget
            reflection_lm=reflection_lm,
            num_threads=num_threads,
            failure_score=0.0,
            perfect_score=1.0,
            seed=seed,
        )

        optimized_pipeline = optimizer.compile(
            pipeline,
            trainset=train_examples,
            valset=eval_examples,
        )

    except ImportError:
        logger.warning("GEPA not available, falling back to MIPROv2 (Multi-step Instruction Proposal)")
        try:
            from dspy.teleprompt import MIPROv2

            # Create blending metric for fallback optimizers too
            if model_rotator is not None:
                def blend_normalized_metric(example, prediction, trace=None):
                    """Normalized metric that rotates through blend models (thread-safe)."""
                    next_lm = model_rotator.next()
                    with dspy.context(lm=next_lm):
                        return normalized_metric(example, prediction, trace)
                fallback_metric = blend_normalized_metric
            else:
                fallback_metric = normalized_metric

            optimizer = MIPROv2(
                metric=fallback_metric,
                auto=optimizer_budget,
                num_threads=num_threads,
                seed=seed,
            )

            optimized_pipeline = optimizer.compile(
                pipeline,
                trainset=train_examples,
                valset=eval_examples,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )

        except Exception as e:
            logger.warning(f"MIPROv2 failed ({e}), falling back to BootstrapFewShot")
            optimizer = dspy.BootstrapFewShot(
                metric=fallback_metric if model_rotator else normalized_metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=2,  # Increased from 1 for better optimization
            )

            optimized_pipeline = optimizer.compile(
                pipeline,
                trainset=train_examples,
            )
    
    # Evaluate optimized pipeline
    logger.info("Evaluating optimized performance...")
    optimized_result = baseline_evaluator(optimized_pipeline)
    optimized_score = _extract_score(optimized_result)
    logger.info(".3f")

    # Also compute raw optimized score
    raw_optimized_result = raw_baseline_evaluator(optimized_pipeline)
    raw_optimized_score = _extract_score(raw_optimized_result)
    logger.info(".1f")

    # Save optimized pipeline
    pipeline_path = output_dir / "optimized_pipeline.json"
    optimized_pipeline.save(str(pipeline_path))
    logger.info(f"Saved optimized pipeline to {pipeline_path}")

    # Calculate improvement
    improvement = optimized_score - baseline_score
    raw_improvement = raw_optimized_score - raw_baseline_score

    if baseline_score != 0:
        improvement_pct = (improvement / abs(baseline_score)) * 100
    else:
        improvement_pct = 0.0

    if raw_baseline_score != 0:
        raw_improvement_pct = (raw_improvement / abs(raw_baseline_score)) * 100
    else:
        raw_improvement_pct = 0.0
    
    # Save optimization results
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "blend_mode": blend_models is not None,
        "blend_models": blend_models if blend_models else None,
        "seed": seed,
        "num_train_examples": num_train_examples,
        "num_eval_examples": num_eval_examples,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "baseline_score_normalized": baseline_score,
        "baseline_score_raw": raw_baseline_score,
        "optimized_score_normalized": optimized_score,
        "optimized_score_raw": raw_optimized_score,
        "improvement_normalized": improvement,
        "improvement_pct_normalized": improvement_pct,
        "improvement_raw": raw_improvement,
        "improvement_pct_raw": raw_improvement_pct,
    }
    
    results_path = output_dir / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    return results


def export_optimized_prompts(
    pipeline_path: Path,
    output_dir: Path,
) -> None:
    """Export optimized pipeline back to markdown prompt files.
    
    Extracts the EVOLVED INSTRUCTIONS from GEPA optimization.
    The key insight is that GEPA stores the improved prompt text in:
    - pipeline_data["clue_giver.generate_clue.predict"]["signature"]["instructions"]
    - pipeline_data["guesser.make_guesses.predict"]["signature"]["instructions"]
    
    NOT in the demos (which are often empty with GEPA).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the optimized pipeline
    with open(pipeline_path, "r") as f:
        pipeline_data = json.load(f)
    
    # Extract evolved clue giver instructions from GEPA
    clue_giver_instructions = None
    clue_giver_key = "clue_giver.generate_clue.predict"
    if clue_giver_key in pipeline_data:
        signature = pipeline_data[clue_giver_key].get("signature", {})
        clue_giver_instructions = signature.get("instructions")
        logger.info(f"Extracted clue_giver instructions: {len(clue_giver_instructions) if clue_giver_instructions else 0} chars")
    
    # Extract evolved guesser instructions from GEPA
    guesser_instructions = None
    guesser_key = "guesser.make_guesses.predict"
    if guesser_key in pipeline_data:
        signature = pipeline_data[guesser_key].get("signature", {})
        guesser_instructions = signature.get("instructions")
        logger.info(f"Extracted guesser instructions: {len(guesser_instructions) if guesser_instructions else 0} chars")
    
    # Write optimized clue giver prompt
    clue_giver_path = output_dir / "clue_giver_optimized.md"
    _write_clue_giver_prompt(clue_giver_path, clue_giver_instructions)
    logger.info(f"Wrote optimized clue giver prompt to {clue_giver_path}")
    
    # Write optimized guesser prompt  
    guesser_path = output_dir / "guesser_optimized.md"
    _write_guesser_prompt(guesser_path, guesser_instructions)
    logger.info(f"Wrote optimized guesser prompt to {guesser_path}")


def _write_clue_giver_prompt(path: Path, evolved_instructions: Optional[str]) -> None:
    """Write clue giver prompt with GEPA-evolved instructions.
    
    If GEPA evolved the instructions, use those directly.
    Otherwise, fall back to the base prompt.
    
    CRITICAL: Always ensures the game template section is present at the end,
    as GEPA may strip it out during optimization.
    """
    # Required template section for clue giver - game won't work without this!
    CLUE_GIVER_TEMPLATE_SECTION = """

---

## CURRENT GAME

**Board (16 words):** {{BOARD}}

**Your Friendly Words (8):** {{FRIENDLY_WORDS}}
**Bystanders:** {{BYSTANDERS}}
**⚠️ ASSASSIN (AVOID AT ALL COSTS):** {{ASSASSIN}}

**Your response MUST include:**
1. Your reasoning
2. CLUE: [single word]
3. NUMBER: [1-8]"""

    if evolved_instructions:
        # GEPA evolved the prompt - use the evolved instructions directly
        prompt = evolved_instructions
        
        # Ensure the template has the required placeholders for the game
        if "{{HEAD_TO_HEAD_CONTEXT}}" not in prompt:
            prompt = "{{HEAD_TO_HEAD_CONTEXT}}\n\n" + prompt
        
        # Clean up any stray ``` from GEPA output
        prompt = prompt.rstrip("`\n ")
        
        # CRITICAL: Ensure the game template section is present
        if "{{FRIENDLY_WORDS}}" not in prompt or "{{ASSASSIN}}" not in prompt:
            prompt = prompt + CLUE_GIVER_TEMPLATE_SECTION
            logger.info("Added missing game template section to clue_giver prompt")
        
        path.write_text(prompt)
        logger.info(f"Wrote GEPA-evolved clue_giver prompt ({len(prompt)} chars)")
    else:
        # Fall back to base prompt if no evolved instructions
        base_prompt_path = Path(__file__).parent.parent / "prompts" / "clue_giver.md"
        base_prompt = base_prompt_path.read_text()
        path.write_text(base_prompt)
        logger.warning("No evolved instructions found, using base prompt")


def _write_guesser_prompt(path: Path, evolved_instructions: Optional[str]) -> None:
    """Write guesser prompt with GEPA-evolved instructions.
    
    If GEPA evolved the instructions, use those directly.
    Otherwise, fall back to the base prompt.
    
    CRITICAL: Always ensures the game template section is present at the end,
    as GEPA may strip it out during optimization.
    """
    # Required template section for guesser - game won't work without this!
    GUESSER_TEMPLATE_SECTION = """

---

## CURRENT GAME

**Available Words (unrevealed):** {{AVAILABLE_WORDS}}

**Board Layout:**
{{BOARD}}

**Current Clue:** {{CLUE}} ({{NUMBER}})

**List your guesses, one word per line. Most confident first. Only choose from available words above.**"""

    if evolved_instructions:
        # GEPA evolved the prompt - use the evolved instructions directly
        prompt = evolved_instructions
        
        # Ensure the template has the required placeholders
        if "{{HEAD_TO_HEAD_CONTEXT}}" not in prompt:
            prompt = "{{HEAD_TO_HEAD_CONTEXT}}\n\n" + prompt
        
        # Clean up any stray ``` from GEPA output
        prompt = prompt.rstrip("`\n ")
        
        # CRITICAL: Ensure the game template section is present
        if "{{AVAILABLE_WORDS}}" not in prompt or "{{CLUE}}" not in prompt:
            prompt = prompt + GUESSER_TEMPLATE_SECTION
            logger.info("Added missing game template section to guesser prompt")
        
        path.write_text(prompt)
        logger.info(f"Wrote GEPA-evolved guesser prompt ({len(prompt)} chars)")
    else:
        # Fall back to base prompt if no evolved instructions
        base_prompt_path = Path(__file__).parent.parent / "prompts" / "guesser.md"
        base_prompt = base_prompt_path.read_text()
        path.write_text(base_prompt)
        logger.warning("No evolved instructions found, using base prompt")


def deploy_optimized_prompts(
    optimized_dir: Path = Path("chainlex/optimized_prompts"),
    backup: bool = True,
) -> Dict[str, str]:
    """Deploy optimized prompts to the main prompts folder.
    
    Copies optimized prompts from the optimization output directory
    to the main chainlex/prompts/ folder, optionally backing up originals.
    
    Args:
        optimized_dir: Directory containing optimized prompts
        backup: Whether to backup original prompts before overwriting
        
    Returns:
        Dict mapping prompt names to their new paths
    """
    import shutil
    
    optimized_dir = Path(optimized_dir)
    prompts_dir = Path(__file__).parent.parent / "prompts"
    
    deployed = {}
    
    # Map of optimized file -> target file
    prompt_mapping = {
        "clue_giver_optimized.md": "clue_giver.md",
        "guesser_optimized.md": "guesser.md",
    }
    
    for optimized_name, target_name in prompt_mapping.items():
        optimized_path = optimized_dir / optimized_name
        target_path = prompts_dir / target_name
        
        if not optimized_path.exists():
            logger.warning(f"Optimized prompt not found: {optimized_path}")
            continue
        
        # Backup original if it exists
        if backup and target_path.exists():
            backup_path = prompts_dir / f"{target_name}.backup"
            shutil.copy(target_path, backup_path)
            logger.info(f"Backed up {target_path} to {backup_path}")
        
        # Copy optimized to target
        shutil.copy(optimized_path, target_path)
        logger.info(f"Deployed {optimized_path} -> {target_path}")
        
        deployed[target_name] = str(target_path)
    
    return deployed


def rollback_prompts() -> Dict[str, str]:
    """Rollback prompts to their backup versions.
    
    Restores original prompts from .backup files created during deployment.
    
    Returns:
        Dict mapping prompt names to their restored paths
    """
    import shutil
    
    prompts_dir = Path(__file__).parent.parent / "prompts"
    
    restored = {}
    
    for prompt_name in ["clue_giver.md", "guesser.md"]:
        backup_path = prompts_dir / f"{prompt_name}.backup"
        target_path = prompts_dir / prompt_name
        
        if backup_path.exists():
            shutil.copy(backup_path, target_path)
            logger.info(f"Restored {backup_path} -> {target_path}")
            restored[prompt_name] = str(target_path)
        else:
            logger.warning(f"No backup found for {prompt_name}")
    
    return restored
