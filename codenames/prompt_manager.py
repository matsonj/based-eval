"""Prompt management for loading and formatting Markdown templates."""

import logging
import re
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PromptHydrationError(Exception):
    """Raised when prompt template hydration fails."""
    pass


class PromptManager:
    """Manages loading and formatting of prompt templates from Markdown files."""

    def __init__(self):
        pass

    def load_prompt(self, prompt_file: str, context: Dict[str, Any]) -> str:
        """Load and format a prompt template with given context."""
        try:
            prompt_path = Path(prompt_file)

            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {prompt_file}, using default")
                return self._get_default_prompt(context)

            template = self._load_with_includes(prompt_path)

            # Format template with context
            formatted_prompt = self._format_template(template, context)

            logger.debug(f"Loaded prompt from {prompt_file}")
            return formatted_prompt

        except PromptHydrationError as e:
            # Re-raise hydration errors to fail fast
            logger.error(f"Prompt hydration failed for {prompt_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_file}: {e}")
            raise PromptHydrationError(f"Failed to load prompt {prompt_file}: {e}")

    def _load_with_includes(self, prompt_path: Path) -> str:
        """Load a prompt file and process any {{include}} directives."""
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Process include directives: {{include:path/to/file.md}}
        include_pattern = r'\{\{include:([^}]+)\}\}'
        
        def replace_include(match):
            include_path = match.group(1).strip()
            # Include paths are relative to prompts directory
            full_include_path = prompt_path.parent / include_path
            
            if not full_include_path.exists():
                logger.warning(f"Include file not found: {full_include_path}")
                return f"<!-- Include not found: {include_path} -->"
            
            try:
                with open(full_include_path, "r", encoding="utf-8") as f:
                    include_content = f.read()
                logger.debug(f"Included content from {full_include_path}")
                return include_content
            except Exception as e:
                logger.error(f"Error loading include {full_include_path}: {e}")
                return f"<!-- Error loading include: {include_path} -->"

        # Replace all include directives
        processed_content = re.sub(include_pattern, replace_include, content)
        return processed_content

    def _format_template(self, template: str, context: Dict[str, Any]) -> str:
        """Format template string with context variables."""
        try:
            # Find all template variables in the template
            template_vars = set(re.findall(r'\{\{([A-Z_]+)\}\}', template))
            
            # Simple template variable replacement
            formatted = template

            for key, value in context.items():
                placeholder = f"{{{{{key.upper()}}}}}"

                if isinstance(value, list):
                    # Format lists nicely
                    if key == "board":
                        # Format board as a grid
                        board_str = self._format_board(value)
                    else:
                        board_str = ", ".join(str(item) for item in value)
                    formatted = formatted.replace(placeholder, board_str)

                elif isinstance(value, dict):
                    # Format dictionaries
                    if key == "identities":
                        dict_str = self._format_identities(value)
                    elif key == "revealed":
                        dict_str = self._format_revealed(value)
                    else:
                        dict_str = str(value)
                    formatted = formatted.replace(placeholder, dict_str)

                else:
                    formatted = formatted.replace(placeholder, str(value))

            # Check for any remaining unhydrated template variables
            remaining_vars = set(re.findall(r'\{\{([A-Z_]+)\}\}', formatted))
            if remaining_vars:
                provided_vars = {key.upper() for key in context.keys()}
                missing_vars = remaining_vars - provided_vars
                
                raise PromptHydrationError(
                    f"Template hydration failed. Missing variables: {missing_vars}. "
                    f"Template expected: {template_vars}. "
                    f"Context provided: {provided_vars}"
                )

            return formatted

        except PromptHydrationError:
            # Re-raise hydration errors
            raise
        except Exception as e:
            logger.error(f"Error formatting template: {e}")
            raise PromptHydrationError(f"Template formatting error: {e}")

    def _format_board(self, board: list) -> str:
        """Format board as a 5x5 grid."""
        if len(board) != 25:
            return ", ".join(board)

        grid_lines = []
        for row in range(5):
            row_items = board[row * 5 : (row + 1) * 5]
            grid_lines.append(" | ".join(f"{item:>12}" for item in row_items))

        return "\n".join(grid_lines)

    def _format_identities(self, identities: dict) -> str:
        """Format identities dictionary."""
        if not identities:
            return "None revealed yet"

        lines = []
        for name, identity in identities.items():
            lines.append(f"{name}: {identity}")

        return "\n".join(lines)

    def _format_revealed(self, revealed: dict) -> str:
        """Format revealed status dictionary."""
        revealed_names = [name for name, is_revealed in revealed.items() if is_revealed]
        if not revealed_names:
            return "None"
        return ", ".join(revealed_names)

    def _get_default_prompt(self, context: Dict[str, Any]) -> str:
        """Generate a default prompt if template file is not available."""
        role = self._infer_role(context)

        if "coach" in role:
            return self._get_default_coach_prompt(context)
        else:
            return self._get_default_player_prompt(context)

    def _infer_role(self, context: Dict[str, Any]) -> str:
        """Infer the role from context."""
        if "identities" in context and context.get("identities"):
            return "coach"
        elif "play" in context:
            return "player"
        else:
            return "coach"  # Default

    def _get_default_coach_prompt(self, context: Dict[str, Any]) -> str:
        """Default coach prompt."""
        team = context.get("team", "red")
        board = context.get("board", [])
        identities = context.get("identities", {})

        prompt = f"""# The Playbook - {team.title()} Team Coach

You are the Coach for the {team.title()} team in The Playbook, a game of strategic deduction.

## Your Mission
Guide your Players to identify all Allied Targets while avoiding:
- Innocent Civilians (waste a shot)
- Enemy Targets (help the other team)
- The Illegal Target (instant loss!)

## Current Board
{self._format_board(board)}

## Secret Intelligence (Only you can see this!)
{self._format_identities(identities)}

## Your Task
Provide a cryptic play and number that will help your Players identify YOUR Allied Targets.

Format your response as:
PLAY: [your cryptic play]
NUMBER: [number of related targets]

Be clever but not too obvious - the enemy might be listening!
"""
        return prompt

    def _get_default_player_prompt(self, context: Dict[str, Any]) -> str:
        """Default player prompt."""
        team = context.get("team", "red")
        board = context.get("board", [])
        play = context.get("play", "")
        number = context.get("number", 1)
        revealed = context.get("revealed", {})

        available_names = [name for name in board if not revealed.get(name, False)]

        prompt = f"""# The Playbook - {team.title()} Team Player

You are a Player for the {team.title()} team in The Playbook.

## Your Mission
Your Coach has given you a cryptic play. Use it to identify Allied Targets.

## Current Board (Available Names)
{', '.join(available_names)}

## Coach's Message
Play: "{play}"
Number: {number}

## Rules
- You can guess up to {number + 1} names (N+1 rule)
- Stop immediately if you're unsure
- Avoid The Mole at all costs!

Respond with your guesses, one per line. You may guess fewer than the maximum allowed.
"""
        return prompt
