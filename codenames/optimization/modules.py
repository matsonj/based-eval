"""DSPy modules for Codenames Spymaster and Operative roles."""

from typing import List, Optional, Literal
import dspy


class SpymasterSignature(dspy.Signature):
    """Give a one-word clue and number to help your operative find your team's agents.
    
    You are the Spymaster in Codenames. You see the full board and know which words belong to your team.
    Give a clue that connects as many of your team's words as possible while avoiding:
    - Enemy agents (opponent's words)
    - Bystanders (neutral words)  
    - The Assassin (instant loss if guessed)
    
    Your clue must be a single word (compound words and proper nouns allowed).
    The number indicates how many of your team's words relate to this clue.
    """
    
    # Inputs
    available_words: str = dspy.InputField(desc="All unrevealed words on the board, comma-separated")
    team_agents: str = dspy.InputField(desc="Your team's agent words (try to get these guessed)")
    enemy_agents: str = dspy.InputField(desc="Enemy team's agent words (avoid these)")
    bystanders: str = dspy.InputField(desc="Neutral bystander words (avoid these)")
    assassin: str = dspy.InputField(desc="The ASSASSIN word - if guessed, your team loses instantly!")
    
    # Outputs
    clue: str = dspy.OutputField(desc="A single word clue (no sentences, no explanations)")
    number: int = dspy.OutputField(desc="How many of YOUR team's words this clue relates to")
    reasoning: str = dspy.OutputField(desc="Brief explanation of which words you're targeting and why this clue is safe")


class OperativeSignature(dspy.Signature):
    """Guess words on the board based on your Spymaster's clue.
    
    You are the Operative in Codenames. You see only the board words (not their identities).
    Your Spymaster gave you a clue and number - find the words that match.
    
    Be strategic:
    - The number tells you how many words to look for
    - You can guess up to number+1 words (plus-one rule)
    - Stop guessing if you're unsure - wrong guesses help the enemy or end the game
    """
    
    # Inputs
    available_words: str = dspy.InputField(desc="Unrevealed words on the board, comma-separated")
    clue: str = dspy.InputField(desc="The one-word clue from your Spymaster")
    number: int = dspy.InputField(desc="How many words the Spymaster says relate to this clue")
    
    # Outputs
    guesses: str = dspy.OutputField(desc="Words to guess, comma-separated, in order of confidence (most confident first)")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why each word matches the clue")


class SpymasterModule(dspy.Module):
    """DSPy module for Codenames Spymaster role.
    
    Takes the full board state and produces a clue + number.
    """
    
    def __init__(self):
        super().__init__()
        self.generate_clue = dspy.ChainOfThought(SpymasterSignature)
    
    def forward(
        self,
        available_words: List[str],
        team_agents: List[str],
        enemy_agents: List[str],
        bystanders: List[str],
        assassin: str,
    ) -> dspy.Prediction:
        """Generate a clue for the given board state.
        
        Args:
            available_words: All unrevealed words
            team_agents: This team's agent words
            enemy_agents: Enemy team's words
            bystanders: Neutral words
            assassin: The assassin word
            
        Returns:
            Prediction with clue, number, and reasoning
        """
        result = self.generate_clue(
            available_words=", ".join(available_words),
            team_agents=", ".join(team_agents),
            enemy_agents=", ".join(enemy_agents),
            bystanders=", ".join(bystanders),
            assassin=assassin,
        )
        
        # Parse the number (handle string outputs)
        try:
            clue_number = int(result.number)
        except (ValueError, TypeError):
            clue_number = 1
        
        return dspy.Prediction(
            clue=result.clue.strip().upper(),
            clue_number=clue_number,
            reasoning=result.reasoning,
        )


class OperativeModule(dspy.Module):
    """DSPy module for Codenames Operative role.
    
    Takes a clue and board state, produces guesses.
    """
    
    def __init__(self):
        super().__init__()
        self.make_guesses = dspy.ChainOfThought(OperativeSignature)
    
    def forward(
        self,
        available_words: List[str],
        clue: str,
        number: int,
    ) -> dspy.Prediction:
        """Generate guesses for the given clue.
        
        Args:
            available_words: Unrevealed words on board
            clue: The spymaster's clue word
            number: How many words the clue targets
            
        Returns:
            Prediction with guesses list and reasoning
        """
        result = self.make_guesses(
            available_words=", ".join(available_words),
            clue=clue,
            number=number,
        )
        
        # Parse guesses from comma-separated string
        raw_guesses = result.guesses.strip()
        guesses = [g.strip().upper() for g in raw_guesses.split(",") if g.strip()]
        
        # Validate guesses are actually on the board
        available_upper = {w.upper() for w in available_words}
        valid_guesses = [g for g in guesses if g in available_upper]
        
        # Enforce plus-one rule (max number + 1 guesses)
        max_guesses = number + 1 if number > 0 else len(available_words)
        valid_guesses = valid_guesses[:max_guesses]
        
        return dspy.Prediction(
            guesses=valid_guesses,
            reasoning=result.reasoning,
        )


class SpymasterModuleSimple(dspy.Module):
    """Simplified Spymaster module using Predict instead of ChainOfThought.
    
    Useful for faster inference or cheaper models.
    """
    
    def __init__(self):
        super().__init__()
        self.generate_clue = dspy.Predict(SpymasterSignature)
    
    def forward(
        self,
        available_words: List[str],
        team_agents: List[str],
        enemy_agents: List[str],
        bystanders: List[str],
        assassin: str,
    ) -> dspy.Prediction:
        result = self.generate_clue(
            available_words=", ".join(available_words),
            team_agents=", ".join(team_agents),
            enemy_agents=", ".join(enemy_agents),
            bystanders=", ".join(bystanders),
            assassin=assassin,
        )
        
        try:
            clue_number = int(result.number)
        except (ValueError, TypeError):
            clue_number = 1
        
        return dspy.Prediction(
            clue=result.clue.strip().upper(),
            clue_number=clue_number,
            reasoning=getattr(result, 'reasoning', ''),
        )


class OperativeModuleSimple(dspy.Module):
    """Simplified Operative module using Predict instead of ChainOfThought."""
    
    def __init__(self):
        super().__init__()
        self.make_guesses = dspy.Predict(OperativeSignature)
    
    def forward(
        self,
        available_words: List[str],
        clue: str,
        number: int,
    ) -> dspy.Prediction:
        result = self.make_guesses(
            available_words=", ".join(available_words),
            clue=clue,
            number=number,
        )
        
        raw_guesses = result.guesses.strip()
        guesses = [g.strip().upper() for g in raw_guesses.split(",") if g.strip()]
        
        available_upper = {w.upper() for w in available_words}
        valid_guesses = [g for g in guesses if g in available_upper]
        
        max_guesses = number + 1 if number > 0 else len(available_words)
        valid_guesses = valid_guesses[:max_guesses]
        
        return dspy.Prediction(
            guesses=valid_guesses,
            reasoning=getattr(result, 'reasoning', ''),
        )

