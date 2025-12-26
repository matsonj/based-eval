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
    
    Returns:
        - ScoreWithFeedback dict when pred_name is provided (GEPA feedback mode)
        - Float score otherwise (standard evaluation mode)
    """
    # Parse board state
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
    
    # Calculate score and identify what went wrong
    score = 0
    correct_count = 0
    feedback_parts = []
    hit_assassin = False
    hit_bystander = False
    
    for i, guess in enumerate(guesses):
        if guess == assassin:
            score += ASSASSIN_PENALTY
            hit_assassin = True
            feedback_parts.append(f"CRITICAL: Guess '{guess}' hit the ASSASSIN! The clue '{clue}' led to the assassin word.")
            break
        elif guess in bystanders:
            score += BYSTANDER_PENALTY
            hit_bystander = True
            feedback_parts.append(f"Guess '{guess}' was a BYSTANDER. The clue '{clue}' accidentally associated with a neutral word.")
            break
        elif guess in friendly_words:
            correct_count += 1
            score += correct_count
    
    # Normalize score to 0-1
    max_score = 36  # 1+2+3+4+5+6+7+8
    if hit_assassin:
        normalized_score = 0.0
    elif hit_bystander:
        normalized_score = 0.1 + 0.1 * (correct_count / 8)  # 0.1 to 0.2
    else:
        normalized_score = 0.5 + 0.5 * (score / max_score)  # 0.5 to 1.0
    
    # If GEPA is asking for predictor-level feedback, return dict
    if pred_name is not None:
        # Generate feedback for GEPA
        if not feedback_parts:
            if correct_count == 0:
                feedback_parts.append(f"The clue '{clue}' ({clue_number}) didn't connect with any guessed words.")
            elif correct_count < clue_number:
                feedback_parts.append(f"The clue '{clue}' ({clue_number}) only led to {correct_count} correct guesses out of {clue_number} intended.")
            else:
                feedback_parts.append(f"Good! The clue '{clue}' ({clue_number}) successfully connected to {correct_count} friendly words.")
        
        # Add specific predictor feedback
        if pred_name == "clue_giver" and hit_assassin:
            feedback_parts.append("IMPROVE CLUE GIVER: Must verify clue does NOT associate with assassin word before giving it.")
        elif pred_name == "guesser" and hit_assassin:
            feedback_parts.append("IMPROVE GUESSER: Be more conservative - stop guessing if uncertain rather than risk hitting assassin.")
        
        feedback = " ".join(feedback_parts)
        return {"score": normalized_score, "feedback": feedback}
    
    # Standard evaluation - just return the score
    return normalized_score


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
    """Generate a random ChainLex-1 board.
    
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
) -> List[dspy.Example]:
    """Generate training examples for DSPy optimization.
    
    Each example is a board state that the pipeline will be evaluated on.
    """
    words = load_words(words_file)
    examples = []
    
    for i in range(num_examples):
        board_data = generate_board(words, seed=base_seed + i)
        
        example = dspy.Example(
            board=", ".join(board_data["board"]),
            friendly_words=", ".join(board_data["friendly_words"]),
            bystanders=", ".join(board_data["bystanders"]),
            assassin=board_data["assassin"],
        ).with_inputs("board", "friendly_words", "bystanders", "assassin")
        
        examples.append(example)
    
    return examples


def run_optimization(
    output_dir: Path,
    model: str = "openai/gpt-4o-mini",
    num_train_examples: int = 30,
    num_eval_examples: int = 10,
    num_threads: int = 4,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    seed: int = 42,
    words_file: str = "inputs/names.yaml",
    clue_giver_prompt: Optional[str] = None,
    guesser_prompt: Optional[str] = None,
    dry_run: bool = False,
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
    
    lm = dspy.LM(
        model=f"openrouter/{model}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False,  # Disable caching to ensure fresh results
    )
    dspy.configure(lm=lm, cache=False)  # Also disable global cache
    
    # Generate training and evaluation examples
    logger.info("Generating training examples...")
    train_examples = create_training_examples(
        num_examples=num_train_examples,
        words_file=words_file,
        base_seed=seed,
    )
    
    logger.info("Generating evaluation examples...")
    eval_examples = create_training_examples(
        num_examples=num_eval_examples,
        words_file=words_file,
        base_seed=seed + 10000,  # Different seed for eval
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
    
    # Evaluate baseline with raw metric (for human-readable scores)
    logger.info("Evaluating baseline performance...")
    baseline_evaluator = dspy.Evaluate(
        devset=eval_examples,
        metric=chainlex_metric,
        num_threads=num_threads,
        display_progress=True,
    )
    baseline_result = baseline_evaluator(pipeline)
    # Extract score from EvaluationResult if needed
    baseline_score = _extract_score(baseline_result)
    logger.info(f"Baseline score: {baseline_score}")
    
    # Try GEPA (Genetic Evolution of Prompts Algorithm) for instruction optimization
    # GEPA uses reflection and textual feedback to evolve better prompts
    logger.info("Running GEPA optimization (evolutionary prompt optimization with feedback)...")
    
    try:
        from dspy.teleprompt import GEPA
        
        # Create reflection LM (should be a strong model)
        reflection_lm = dspy.LM(
            model=f"openrouter/{model}",  # Use same model for reflection
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            cache=False,
        )
        
        optimizer = GEPA(
            metric=gepa_feedback_metric,  # Use feedback-aware metric
            auto="light",  # Budget: "light", "medium", or "heavy"
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
        
    except Exception as e:
        # Fall back to BootstrapFewShot if GEPA fails
        logger.warning(f"GEPA failed ({e}), falling back to BootstrapFewShot")
        optimizer = dspy.BootstrapFewShot(
            metric=normalized_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=1,
        )
        
        optimized_pipeline = optimizer.compile(
            pipeline,
            trainset=train_examples,
        )
    
    # Evaluate optimized pipeline
    logger.info("Evaluating optimized performance...")
    optimized_result = baseline_evaluator(optimized_pipeline)
    optimized_score = _extract_score(optimized_result)
    logger.info(f"Optimized score: {optimized_score}")
    
    # Save optimized pipeline
    pipeline_path = output_dir / "optimized_pipeline.json"
    optimized_pipeline.save(str(pipeline_path))
    logger.info(f"Saved optimized pipeline to {pipeline_path}")
    
    # Calculate improvement
    improvement = optimized_score - baseline_score
    if baseline_score != 0:
        improvement_pct = (improvement / abs(baseline_score)) * 100
    else:
        improvement_pct = 0.0
    
    # Save optimization results
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "seed": seed,
        "num_train_examples": num_train_examples,
        "num_eval_examples": num_eval_examples,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
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
    
    Extracts the optimized instructions and few-shot examples from the
    DSPy pipeline and writes them as markdown files compatible with
    the game's template system.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the optimized pipeline
    with open(pipeline_path, "r") as f:
        pipeline_data = json.load(f)
    
    # Extract and format clue giver prompt
    clue_giver_demos = []
    if "clue_giver.generate_clue" in pipeline_data:
        demos = pipeline_data["clue_giver.generate_clue"].get("demos", [])
        for demo in demos:
            if "clue" in demo and "reasoning" in demo:
                clue_giver_demos.append(demo)
    
    # Extract and format guesser prompt
    guesser_demos = []
    if "guesser.make_guesses" in pipeline_data:
        demos = pipeline_data["guesser.make_guesses"].get("demos", [])
        for demo in demos:
            if "guesses" in demo and "reasoning" in demo:
                guesser_demos.append(demo)
    
    # Write optimized clue giver prompt
    clue_giver_path = output_dir / "clue_giver_optimized.md"
    _write_clue_giver_prompt(clue_giver_path, clue_giver_demos)
    logger.info(f"Wrote optimized clue giver prompt to {clue_giver_path}")
    
    # Write optimized guesser prompt  
    guesser_path = output_dir / "guesser_optimized.md"
    _write_guesser_prompt(guesser_path, guesser_demos)
    logger.info(f"Wrote optimized guesser prompt to {guesser_path}")


def _write_clue_giver_prompt(path: Path, demos: List[Dict]) -> None:
    """Write clue giver prompt with optimized few-shot examples."""
    # Load base prompt
    base_prompt_path = Path(__file__).parent.parent / "prompts" / "clue_giver.md"
    base_prompt = base_prompt_path.read_text()
    
    # If we have demos, add them as examples
    if demos:
        examples_section = "\n\n## Optimized Examples\n"
        for i, demo in enumerate(demos, 1):
            examples_section += f"\n**Example {i}:**\n"
            if "board" in demo:
                examples_section += f"Board: {demo['board']}\n"
            if "friendly_words" in demo:
                examples_section += f"Friendly: {demo['friendly_words']}\n"
            if "bystanders" in demo:
                examples_section += f"Bystanders: {demo['bystanders']}\n"
            if "assassin" in demo:
                examples_section += f"Assassin: {demo['assassin']}\n"
            if "reasoning" in demo:
                examples_section += f"Reasoning: {demo['reasoning']}\n"
            if "clue" in demo and "number" in demo:
                examples_section += f"→ CLUE: {demo['clue']}, NUMBER: {demo['number']}\n"
        
        # Insert before the CURRENT GAME section
        if "## CURRENT GAME" in base_prompt:
            base_prompt = base_prompt.replace(
                "## CURRENT GAME",
                f"{examples_section}\n---\n\n## CURRENT GAME"
            )
    
    path.write_text(base_prompt)


def _write_guesser_prompt(path: Path, demos: List[Dict]) -> None:
    """Write guesser prompt with optimized few-shot examples."""
    # Load base prompt
    base_prompt_path = Path(__file__).parent.parent / "prompts" / "guesser.md"
    base_prompt = base_prompt_path.read_text()
    
    # If we have demos, add them as examples
    if demos:
        examples_section = "\n\n## Optimized Examples\n"
        for i, demo in enumerate(demos, 1):
            examples_section += f"\n**Example {i}:**\n"
            if "board" in demo:
                examples_section += f"Board: {demo['board']}\n"
            if "clue" in demo:
                examples_section += f"Clue: {demo['clue']}"
            if "number" in demo:
                examples_section += f" ({demo['number']})\n"
            if "reasoning" in demo:
                examples_section += f"Reasoning: {demo['reasoning']}\n"
            if "guesses" in demo:
                guesses = demo['guesses']
                if isinstance(guesses, list):
                    guesses = ", ".join(guesses)
                examples_section += f"→ Guesses: {guesses}\n"
        
        # Insert before the CURRENT GAME section
        if "## CURRENT GAME" in base_prompt:
            base_prompt = base_prompt.replace(
                "## CURRENT GAME",
                f"{examples_section}\n---\n\n## CURRENT GAME"
            )
    
    path.write_text(base_prompt)


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
