"""Main script for running DSPy optimization on Codenames prompts.

Usage:
    uv run python -m codenames.optimization.optimize --help
    uv run python -m codenames.optimization.optimize --logs-dir logs --output optimized_prompts/
"""

import json
import os
from pathlib import Path
from typing import Optional
import argparse

import dspy

from codenames.optimization.data_extractor import (
    extract_training_examples,
    summarize_examples,
    ClueExample,
)
from codenames.optimization.pipeline import (
    CodenamesPipeline,
    CodenamesTurnDataset,
    pipeline_metric,
)


def setup_dspy_lm(
    model: str = "openai/gpt-4o-mini",
) -> dspy.LM:
    """Configure DSPy to use OpenRouter as the LM provider.
    
    DSPy uses LiteLLM which requires the 'openrouter/' prefix for OpenRouter models.
    
    Args:
        model: Model identifier for OpenRouter (e.g., "google/gemini-3-flash-preview")
        
    Returns:
        Configured DSPy LM instance
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # LiteLLM requires 'openrouter/' prefix for OpenRouter models
    # e.g., "google/gemini-3-flash-preview" -> "openrouter/google/gemini-3-flash-preview"
    if not model.startswith("openrouter/"):
        litellm_model = f"openrouter/{model}"
    else:
        litellm_model = model
    
    lm = dspy.LM(
        model=litellm_model,
        api_key=api_key,
        max_tokens=1000,
    )
    
    dspy.configure(lm=lm)
    return lm


def split_dataset(
    examples: list[ClueExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[ClueExample], list[ClueExample]]:
    """Split examples into train and validation sets.
    
    Args:
        examples: All extracted examples
        train_ratio: Fraction to use for training
        seed: Random seed for reproducibility
        
    Returns:
        (train_examples, val_examples)
    """
    import random
    random.seed(seed)
    
    shuffled = list(examples)
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def run_optimization(
    logs_dir: Path,
    output_dir: Path,
    model: str = "openai/gpt-4o-mini",
    num_threads: int = 4,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 8,
    num_candidates: int = 10,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """Run DSPy optimization on Codenames prompts.
    
    Args:
        logs_dir: Path to logs directory with game_metadata files
        output_dir: Directory to save optimized prompts
        model: Model to use for optimization
        num_threads: Number of parallel evaluation threads
        max_bootstrapped_demos: Max few-shot examples to bootstrap
        max_labeled_demos: Max labeled examples for training
        num_candidates: Number of instruction candidates to try
        seed: Random seed
        dry_run: If True, just test data loading without optimization
        
    Returns:
        Dict with optimization results
    """
    print(f"📂 Loading training data from {logs_dir}...")
    
    # Extract training examples
    all_examples = extract_training_examples(
        logs_dir=logs_dir,
        min_clue_number=1,
        exclude_assassin_hits=False,  # Keep assassin examples for learning
    )
    
    if not all_examples:
        print("❌ No training examples found. Check your logs directory.")
        return {"error": "No training examples"}
    
    # Summary
    summary = summarize_examples(all_examples)
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {summary['count']}")
    print(f"   Average score: {summary['avg_score']:.2f}")
    print(f"   Assassin hit rate: {summary['assassin_rate']:.2%}")
    print(f"   Average clue number: {summary['avg_clue_number']:.1f}")
    print(f"   Average correct guesses: {summary['avg_correct_guesses']:.2f}")
    
    # Split into train/val
    train_examples, val_examples = split_dataset(all_examples, train_ratio=0.8, seed=seed)
    print(f"\n📦 Train/Val split: {len(train_examples)} / {len(val_examples)}")
    
    if dry_run:
        print("\n🔍 Dry run mode - testing with 3 examples...")
        train_examples = train_examples[:3]
        val_examples = val_examples[:1] if val_examples else train_examples[:1]
    
    # Convert to DSPy format
    train_dataset = CodenamesTurnDataset(train_examples)
    val_dataset = CodenamesTurnDataset(val_examples)
    
    print(f"\n🔧 Configuring DSPy with model: {model}")
    lm = setup_dspy_lm(model=model)
    
    # Create the pipeline
    pipeline = CodenamesPipeline(use_chain_of_thought=True)
    
    # Evaluate baseline performance
    print("\n📈 Evaluating baseline performance...")
    from dspy.evaluate import Evaluate
    
    evaluator = Evaluate(
        devset=list(val_dataset),
        metric=pipeline_metric,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,
    )
    
    baseline_result = evaluator(pipeline)
    baseline_score = baseline_result.score if hasattr(baseline_result, 'score') else float(baseline_result)
    print(f"   Baseline average score: {baseline_score:.2f}")
    
    if dry_run:
        print("\n✅ Dry run complete. Data loading and baseline evaluation working.")
        return {
            "mode": "dry_run",
            "num_examples": len(all_examples),
            "baseline_score": baseline_score,
        }
    
    # Run optimization
    print("\n🚀 Running DSPy optimization...")
    print(f"   Optimizer: BootstrapFewShotWithRandomSearch")
    print(f"   Max bootstrapped demos: {max_bootstrapped_demos}")
    print(f"   Max labeled demos: {max_labeled_demos}")
    print(f"   Num candidates: {num_candidates}")
    
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch
    
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=pipeline_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_candidate_programs=num_candidates,
        num_threads=num_threads,
    )
    
    optimized_pipeline = optimizer.compile(
        pipeline,
        trainset=list(train_dataset),
        valset=list(val_dataset),
    )
    
    # Evaluate optimized performance
    print("\n📈 Evaluating optimized performance...")
    optimized_result = evaluator(optimized_pipeline)
    optimized_score = optimized_result.score if hasattr(optimized_result, 'score') else float(optimized_result)
    print(f"   Optimized average score: {optimized_score:.2f}")
    print(f"   Improvement: {optimized_score - baseline_score:+.2f}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the optimized pipeline state
    optimized_pipeline.save(output_dir / "optimized_pipeline.json")
    print(f"\n💾 Saved optimized pipeline to {output_dir / 'optimized_pipeline.json'}")
    
    # Save results summary
    results = {
        "model": model,
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - baseline_score,
        "optimizer_config": {
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "num_candidates": num_candidates,
        },
    }
    
    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Optimization complete!")
    print(f"   Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Codenames prompts using DSPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing game_metadata_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_prompts"),
        help="Directory to save optimized prompts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use for optimization (OpenRouter model ID)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel evaluation threads",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Maximum number of few-shot demonstrations",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="Number of candidate programs to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just test data loading without running optimization",
    )
    
    args = parser.parse_args()
    
    run_optimization(
        logs_dir=args.logs_dir,
        output_dir=args.output,
        model=args.model,
        num_threads=args.threads,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos * 2,
        num_candidates=args.num_candidates,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

