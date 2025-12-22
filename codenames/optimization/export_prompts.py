"""Export optimized DSPy pipeline to game-compatible markdown prompts."""

import json
from pathlib import Path
from typing import Dict, Any, List


def load_optimized_pipeline(pipeline_path: Path) -> Dict[str, Any]:
    """Load the optimized pipeline JSON."""
    with open(pipeline_path) as f:
        return json.load(f)


def format_demo_for_spymaster(demo: Dict[str, Any]) -> str:
    """Format a single demo for the spymaster prompt."""
    # Handle both string and list formats
    available = demo.get("available_words", "")
    if isinstance(available, list):
        available = ", ".join(available)
    
    team = demo.get("team_agents", "")
    if isinstance(team, list):
        team = ", ".join(team)
    
    enemy = demo.get("enemy_agents", "")
    if isinstance(enemy, list):
        enemy = ", ".join(enemy)
    
    bystanders = demo.get("bystanders", "")
    if isinstance(bystanders, list):
        bystanders = ", ".join(bystanders)
    
    assassin = demo.get("assassin", "")
    
    # Get clue and number - may be in different formats
    clue = demo.get("clue", demo.get("expected_clue", ""))
    number = demo.get("number", demo.get("expected_number", 1))
    
    # Get reasoning if available
    reasoning = demo.get("reasoning", "")
    
    lines = [
        f"Board: {available}",
        f"Your Agents: {team}",
        f"Enemy Agents: {enemy}",
        f"Bystanders: {bystanders}",
        f"Assassin: {assassin}",
    ]
    
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    
    lines.append(f"→ CLUE: {clue}, NUMBER: {number}")
    
    return "\n".join(lines)


def format_demo_for_operative(demo: Dict[str, Any]) -> str:
    """Format a single demo for the operative prompt."""
    # Handle both string and list formats
    available = demo.get("available_words", "")
    if isinstance(available, list):
        available = ", ".join(available)
    
    clue = demo.get("clue", demo.get("expected_clue", ""))
    number = demo.get("number", demo.get("expected_number", 1))
    
    # Get guesses - may be in different formats
    guesses = demo.get("guesses", demo.get("expected_guesses", []))
    if isinstance(guesses, list):
        guesses = ", ".join(guesses)
    
    reasoning = demo.get("reasoning", "")
    
    lines = [
        f"Board: {available}",
        f"Clue: {clue} ({number})",
    ]
    
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    
    lines.append(f"→ Guesses: {guesses}")
    
    return "\n".join(lines)


def export_spymaster_prompt(
    demos: List[Dict[str, Any]],
    team: str = "blue",
) -> str:
    """Generate the spymaster prompt with demos."""
    
    # Team-specific variable mapping
    if team == "blue":
        team_var = "{{BLUE_AGENTS}}"
        enemy_var = "{{RED_AGENTS}}"
        remaining_var = "{{BLUE_REMAINING}}"
        enemy_remaining_var = "{{RED_REMAINING}}"
    else:
        team_var = "{{RED_AGENTS}}"
        enemy_var = "{{BLUE_AGENTS}}"
        remaining_var = "{{RED_REMAINING}}"
        enemy_remaining_var = "{{BLUE_REMAINING}}"
    
    prompt = """You are the Spymaster in Codenames. Give a one-word clue and number to help your operative find your team's agents.

Give a clue that connects as many of your team's words as possible while avoiding:
- Enemy agents (opponent's words)
- Bystanders (neutral words)
- The Assassin (instant loss if guessed)

**CRITICAL RULES:**
- Your clue must be a single word (compound words and proper nouns allowed)
- **NEVER repeat a clue that appears in the Clue History below**
- The number indicates how many of YOUR team's remaining words relate to this clue
- Only consider words that are NOT yet revealed

## Examples
"""
    
    # Add demos
    for i, demo in enumerate(demos[:4], 1):  # Limit to 4 demos
        prompt += f"\n**Example {i}:**\n"
        prompt += format_demo_for_spymaster(demo)
        prompt += "\n"
    
    # Add actual task section
    prompt += f"""
---

## CURRENT GAME STATE

**Clue History (DO NOT REPEAT THESE CLUES):**
{{{{CLUE_HISTORY}}}}

**Already Revealed Words (ignore these):**
{{{{REVEALED}}}}

**Remaining Agents:** Your team: {remaining_var} | Enemy: {enemy_remaining_var}

---

## YOUR TASK

Available Words (unrevealed): {{{{BOARD}}}}
Your Agents: {team_var}
Enemy Agents: {enemy_var}
Bystanders: {{{{BYSTANDERS}}}}
Assassin: {{{{ASSASSIN}}}}

**Respond with EXACTLY this format (no other text):**
```
CLUE: [single word - MUST be different from all clues in history]
NUMBER: [count of your remaining agents this relates to]
```
"""
    
    return prompt


def export_operative_prompt(
    demos: List[Dict[str, Any]],
    team: str = "blue",
) -> str:
    """Generate the operative prompt with demos."""
    
    prompt = """You are the Operative in Codenames. Guess words on the board based on your Spymaster's clue.

You see only the board words (not their identities). Your Spymaster gave you a clue and number - find the words that match.

Be strategic:
- The number tells you how many words to look for
- You can guess up to number+1 words (plus-one rule)
- Stop guessing if you're unsure - wrong guesses help the enemy or end the game
- Check the clue history for context from previous turns

## Examples
"""
    
    # Add demos
    for i, demo in enumerate(demos[:4], 1):  # Limit to 4 demos
        prompt += f"\n**Example {i}:**\n"
        prompt += format_demo_for_operative(demo)
        prompt += "\n"
    
    # Add actual task section
    prompt += """
---

## CURRENT GAME STATE

**Clue History:**
{{CLUE_HISTORY}}

---

## YOUR TASK

Available Words (unrevealed): {{BOARD}}
Current Clue: {{CLUE}} ({{NUMBER}})

**List your guesses, one word per line. Most confident first. Only choose words from the available words above.**
"""
    
    return prompt


def export_prompts_from_pipeline(
    pipeline_path: Path,
    output_dir: Path,
) -> None:
    """Export optimized pipeline to game-compatible prompt files.
    
    Args:
        pipeline_path: Path to optimized_pipeline.json
        output_dir: Directory to write prompt files (usually codenames/prompts/)
    """
    pipeline = load_optimized_pipeline(pipeline_path)
    
    # Extract demos from the pipeline
    spymaster_data = pipeline.get("spymaster.generate_clue.predict", {})
    operative_data = pipeline.get("operative.make_guesses.predict", {})
    
    spymaster_demos = spymaster_data.get("demos", [])
    operative_demos = operative_data.get("demos", [])
    
    # Filter to only augmented demos (the bootstrapped ones with reasoning)
    spymaster_augmented = [d for d in spymaster_demos if d.get("augmented", False)]
    operative_augmented = [d for d in operative_demos if d.get("augmented", False)]
    
    # If not enough augmented, use all demos
    if len(spymaster_augmented) < 2:
        spymaster_augmented = spymaster_demos
    if len(operative_augmented) < 2:
        operative_augmented = operative_demos
    
    print(f"Found {len(spymaster_augmented)} spymaster demos")
    print(f"Found {len(operative_augmented)} operative demos")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate prompts for both teams
    for team in ["blue", "red"]:
        # Spymaster prompt
        spymaster_prompt = export_spymaster_prompt(spymaster_augmented, team=team)
        spymaster_path = output_dir / f"{team}_spymaster.md"
        with open(spymaster_path, "w") as f:
            f.write(spymaster_prompt)
        print(f"✅ Wrote {spymaster_path}")
        
        # Operative prompt
        operative_prompt = export_operative_prompt(operative_augmented, team=team)
        operative_path = output_dir / f"{team}_operative.md"
        with open(operative_path, "w") as f:
            f.write(operative_prompt)
        print(f"✅ Wrote {operative_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export optimized DSPy pipeline to prompts")
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("optimized_prompts/optimized_pipeline.json"),
        help="Path to optimized_pipeline.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("codenames/prompts"),
        help="Directory to write prompt files",
    )
    
    args = parser.parse_args()
    
    if not args.pipeline.exists():
        print(f"❌ Pipeline file not found: {args.pipeline}")
        return
    
    export_prompts_from_pipeline(args.pipeline, args.output)
    print("\n✅ Export complete!")


if __name__ == "__main__":
    main()

