"""DSPy modules for ChainLex-1 optimization.

Loads prompts from markdown files and defines signatures that can be optimized using DSPy.
"""

from pathlib import Path
from typing import Optional

import dspy


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt from the chainlex/prompts/ folder."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompt_path = prompts_dir / f"{prompt_name}.md"
    
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def _get_clue_giver_instructions() -> str:
    """Load clue giver instructions from markdown file."""
    return _load_prompt("clue_giver")


def _get_guesser_instructions() -> str:
    """Load guesser instructions from markdown file."""
    return _load_prompt("guesser")


class ClueGiverSignature(dspy.Signature):
    """Generate a clue for ChainLex-1. Instructions loaded from chainlex/prompts/clue_giver.md"""
    
    board: str = dspy.InputField(desc="All 16 words on the board")
    friendly_words: str = dspy.InputField(desc="Your target words")
    bystanders: str = dspy.InputField(desc="Neutral words (-5 penalty)")
    assassin: str = dspy.InputField(desc="Word to avoid (-999 penalty)")
    
    reasoning: str = dspy.OutputField(desc="Your reasoning")
    clue: str = dspy.OutputField(desc="Single word clue")
    number: int = dspy.OutputField(desc="Number of words (1-8)")


class GuesserSignature(dspy.Signature):
    """Make guesses for ChainLex-1. Instructions loaded from chainlex/prompts/guesser.md"""
    
    board: str = dspy.InputField(desc="All 16 words on the board")
    clue: str = dspy.InputField(desc="The clue word")
    number: int = dspy.InputField(desc="Number indicated by clue giver")
    
    reasoning: str = dspy.OutputField(desc="Your reasoning")
    guesses: str = dspy.OutputField(desc="Comma-separated guesses, most confident first")


class ClueGiver(dspy.Module):
    """DSPy module for generating clues in ChainLex-1.
    
    Loads instructions from chainlex/prompts/clue_giver.md
    """
    
    def __init__(self, prompt_file: Optional[str] = None):
        super().__init__()
        
        # Load instructions from markdown file
        if prompt_file:
            instructions = Path(prompt_file).read_text()
        else:
            instructions = _get_clue_giver_instructions()
        
        # Create signature with loaded instructions
        self.generate_clue = dspy.ChainOfThought(
            ClueGiverSignature.with_instructions(instructions)
        )
    
    def forward(self, board: str, friendly_words: str, bystanders: str, assassin: str):
        result = self.generate_clue(
            board=board,
            friendly_words=friendly_words,
            bystanders=bystanders,
            assassin=assassin,
        )
        
        # Validate and clean the number
        try:
            number = int(result.number)
            number = max(1, min(8, number))  # Clamp to 1-8
        except (ValueError, TypeError):
            number = 3  # Default fallback
        
        # Clean the clue (remove quotes, spaces)
        clue = str(result.clue).strip().strip('"\'').upper()
        if ' ' in clue:
            clue = clue.split()[0]  # Take first word if multiple
        
        return dspy.Prediction(
            reasoning=result.reasoning,
            clue=clue,
            number=number,
        )


class Guesser(dspy.Module):
    """DSPy module for making guesses in ChainLex-1.
    
    Loads instructions from chainlex/prompts/guesser.md
    """
    
    def __init__(self, prompt_file: Optional[str] = None):
        super().__init__()
        
        # Load instructions from markdown file
        if prompt_file:
            instructions = Path(prompt_file).read_text()
        else:
            instructions = _get_guesser_instructions()
        
        self.make_guesses = dspy.ChainOfThought(
            GuesserSignature.with_instructions(instructions)
        )
    
    def forward(self, board: str, clue: str, number: int):
        result = self.make_guesses(
            board=board,
            clue=clue,
            number=number,
        )
        
        # Parse guesses into a list
        guesses_str = str(result.guesses).strip()
        board_words = set(word.strip().upper() for word in board.split(','))
        
        guesses = []
        for guess in guesses_str.replace(';', ',').split(','):
            clean_guess = guess.strip().strip('"\'').upper()
            # Only include valid board words
            if clean_guess in board_words and clean_guess not in guesses:
                guesses.append(clean_guess)
        
        # Limit to number + 1 (plus-one rule)
        max_guesses = min(number + 1, 9)
        guesses = guesses[:max_guesses]
        
        return dspy.Prediction(
            reasoning=result.reasoning,
            guesses=guesses,
        )


class ChainLexPipeline(dspy.Module):
    """Complete ChainLex-1 pipeline: ClueGiver -> Guesser.
    
    Loads prompts from chainlex/prompts/ folder.
    """
    
    def __init__(self, clue_giver_prompt: Optional[str] = None, guesser_prompt: Optional[str] = None):
        super().__init__()
        self.clue_giver = ClueGiver(prompt_file=clue_giver_prompt)
        self.guesser = Guesser(prompt_file=guesser_prompt)
    
    def forward(self, board: str, friendly_words: str, bystanders: str, assassin: str):
        """Run the full pipeline and return clue + guesses."""
        # Generate clue
        clue_result = self.clue_giver(
            board=board,
            friendly_words=friendly_words,
            bystanders=bystanders,
            assassin=assassin,
        )
        
        # Make guesses based on the clue
        guess_result = self.guesser(
            board=board,
            clue=clue_result.clue,
            number=clue_result.number,
        )
        
        return dspy.Prediction(
            clue=clue_result.clue,
            clue_number=clue_result.number,
            clue_reasoning=clue_result.reasoning,
            guesses=guess_result.guesses,
            guess_reasoning=guess_result.reasoning,
        )

