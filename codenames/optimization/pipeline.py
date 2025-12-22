"""Joint optimization pipeline for Codenames Spymaster + Operative."""

from typing import List, Optional, Dict, Any
import dspy

from codenames.optimization.modules import SpymasterModule, OperativeModule
from codenames.optimization.metrics import codenames_score, CodenamesTotalScore
from codenames.optimization.referee import RuleBasedReferee, ValidationResult


# Penalty score for invalid clues
# An invalid clue ends your turn and reveals one opponent agent
# This is roughly equivalent to guessing an enemy agent (-1.0)
INVALID_CLUE_PENALTY = -1.0


class CodenamesPipeline(dspy.Module):
    """Joint Spymaster + Operative pipeline for end-to-end optimization.
    
    This pipeline connects:
    1. Spymaster: Sees full board → produces clue + number
    2. Operative: Sees clue + available words → produces guesses
    
    By optimizing this pipeline jointly, DSPy can learn prompts that work
    well together - the spymaster learns to give clues the operative understands,
    and the operative learns to interpret clues effectively.
    """
    
    def __init__(self, use_chain_of_thought: bool = True, validate_clues: bool = True):
        super().__init__()
        
        if use_chain_of_thought:
            self.spymaster = SpymasterModule()
            self.operative = OperativeModule()
        else:
            from codenames.optimization.modules import (
                SpymasterModuleSimple, 
                OperativeModuleSimple,
            )
            self.spymaster = SpymasterModuleSimple()
            self.operative = OperativeModuleSimple()
        
        # Rule-based referee for clue validation (no LLM calls)
        self.validate_clues = validate_clues
        self.referee = RuleBasedReferee() if validate_clues else None
    
    def forward(
        self,
        available_words: List[str],
        team_agents: List[str],
        enemy_agents: List[str],
        bystanders: List[str],
        assassin: str,
    ) -> dspy.Prediction:
        """Run the full pipeline: Spymaster → Referee → Operative.
        
        Args:
            available_words: All unrevealed words on board
            team_agents: This team's agent words
            enemy_agents: Enemy team's words
            bystanders: Neutral words
            assassin: The assassin word
            
        Returns:
            Prediction with clue, number, guesses, and score
        """
        # Step 1: Spymaster generates clue
        spymaster_result = self.spymaster(
            available_words=available_words,
            team_agents=team_agents,
            enemy_agents=enemy_agents,
            bystanders=bystanders,
            assassin=assassin,
        )
        
        clue = spymaster_result.clue
        clue_number = spymaster_result.clue_number
        
        # Step 2: Referee validates clue (rule-based, no LLM cost)
        clue_valid = True
        clue_invalid_reason = ""
        
        if self.validate_clues and self.referee:
            validation_result = self.referee.validate(clue, available_words)
            clue_valid = validation_result.is_valid
            clue_invalid_reason = validation_result.reason
            
            if not clue_valid:
                # Invalid clue - return penalty score, no guesses
                return dspy.Prediction(
                    clue=clue,
                    clue_number=clue_number,
                    guesses=[],
                    spymaster_reasoning=spymaster_result.reasoning,
                    operative_reasoning="",
                    score=INVALID_CLUE_PENALTY,
                    score_details=None,
                    clue_valid=False,
                    clue_invalid_reason=clue_invalid_reason,
                )
        
        # Step 3: Operative makes guesses based on clue
        operative_result = self.operative(
            available_words=available_words,
            clue=clue,
            number=clue_number,
        )
        
        guesses = operative_result.guesses
        
        # Step 4: Score the result
        score_result = codenames_score(
            guesses=guesses,
            team_agents=team_agents,
            enemy_agents=enemy_agents,
            bystanders=bystanders,
            assassin=assassin,
            clue_number=clue_number,
        )
        
        return dspy.Prediction(
            clue=clue,
            clue_number=clue_number,
            guesses=guesses,
            spymaster_reasoning=spymaster_result.reasoning,
            operative_reasoning=operative_result.reasoning,
            score=score_result.raw_score,
            score_details=score_result,
            clue_valid=True,
            clue_invalid_reason="",
        )


class CodenamesTurnDataset:
    """DSPy-compatible dataset for Codenames turn examples.
    
    Wraps a list of ClueExample objects into DSPy Example format.
    """
    
    def __init__(self, examples: List["ClueExample"]):
        """Initialize with ClueExample objects from data_extractor."""
        from codenames.optimization.data_extractor import ClueExample
        
        self.examples = []
        for ex in examples:
            # Filter to unrevealed words
            available = [w for w in ex.board_words if w not in ex.revealed_words]
            team_agents = [w for w in ex.team_agents if w not in ex.revealed_words]
            enemy_agents = [w for w in ex.enemy_agents if w not in ex.revealed_words]
            bystanders = [w for w in ex.bystanders if w not in ex.revealed_words]
            
            dspy_example = dspy.Example(
                # Inputs
                available_words=available,
                team_agents=team_agents,
                enemy_agents=enemy_agents,
                bystanders=bystanders,
                assassin=ex.assassin,
                # Ground truth (for evaluation)
                expected_clue=ex.clue,
                expected_number=ex.clue_number,
                expected_guesses=ex.guesses,
                expected_score=ex.total_score,
            ).with_inputs(
                "available_words",
                "team_agents", 
                "enemy_agents",
                "bystanders",
                "assassin",
            )
            
            self.examples.append(dspy_example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __iter__(self):
        return iter(self.examples)


def pipeline_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Metric function for evaluating the joint pipeline.
    
    This is the main optimization target. Higher scores are better.
    
    Scoring:
    - +1 per correct guess (team agent)
    - -0.5 per bystander guess
    - -1 per enemy guess
    - -999 per assassin guess
    """
    score_result = codenames_score(
        guesses=prediction.guesses,
        team_agents=example.team_agents,
        enemy_agents=example.enemy_agents,
        bystanders=example.bystanders,
        assassin=example.assassin,
        clue_number=prediction.clue_number,
    )
    
    return score_result.raw_score


def pipeline_metric_normalized(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Normalized metric that rewards ambitious clues that succeed.
    
    Divides the raw score by clue_number to reward higher-number clues
    that actually get their words guessed.
    """
    score_result = codenames_score(
        guesses=prediction.guesses,
        team_agents=example.team_agents,
        enemy_agents=example.enemy_agents,
        bystanders=example.bystanders,
        assassin=example.assassin,
        clue_number=prediction.clue_number,
    )
    
    return score_result.normalized_score

