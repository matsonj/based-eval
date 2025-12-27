"""DSPy-based prompt optimization for Codenames."""

from codenames.optimization.data_extractor import (
    extract_training_examples,
    ClueExample,
    GuessOutcome,
)
from codenames.optimization.metrics import codenames_score, CodenamesTotalScore
from codenames.optimization.modules import SpymasterModule, OperativeModule
from codenames.optimization.pipeline import CodenamesPipeline
from codenames.optimization.referee import (
    RuleBasedReferee,
    ValidationResult,
    validate_clue,
    validate_clue_quick,
)

__all__ = [
    "extract_training_examples",
    "ClueExample", 
    "GuessOutcome",
    "codenames_score",
    "CodenamesTotalScore",
    "SpymasterModule",
    "OperativeModule",
    "CodenamesPipeline",
    "RuleBasedReferee",
    "ValidationResult",
    "validate_clue",
    "validate_clue_quick",
]

