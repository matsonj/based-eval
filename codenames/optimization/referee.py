"""Rule-based deterministic referee for Codenames clue validation.

This module provides fast, free, and deterministic clue validation
without needing LLM calls. Used in DSPy optimization to penalize
invalid clues during training.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set


# Common English suffixes for grammatical variant detection
VERB_SUFFIXES = ['ing', 'ed', 's', 'es', 'er', 'ers', 'tion', 'sion']
NOUN_SUFFIXES = ['s', 'es', 'ies', 'er', 'ers', 'or', 'ors', 'ist', 'ists', 'ment', 'ness', 'ity']
ADJ_SUFFIXES = ['er', 'est', 'ly', 'ful', 'less', 'ness', 'ish', 'ive', 'ous', 'al']

# Patterns that indicate letter count references
LETTER_COUNT_PATTERNS = [
    r'\b\d+[-\s]?letters?\b',           # "5 letters", "5-letter"
    r'\b(one|two|three|four|five|six|seven|eight|nine|ten)[-\s]?letters?\b',
    r'\bletter[-\s]?count\b',
    r'\bsame[-\s]?length\b',
]

# Patterns that indicate board position references  
POSITION_PATTERNS = [
    r'\b(top|bottom|left|right|middle|center|corner)\s*(row|column)?\b',
    r'\b(first|second|third|fourth|fifth|last)\s*(row|column)?\b',
    r'\brow\s*\d+\b',
    r'\bcolumn\s*\d+\b',
    r'\bposition\b',
    r'\blocation\b',
    r'\b(row|column)\s*(one|two|three|four|five)\b',
]


@dataclass
class ValidationResult:
    """Result of clue validation."""
    is_valid: bool
    reason: str
    rule_violated: Optional[str] = None  # Which of the 5 rules was violated
    related_word: Optional[str] = None   # Board word that caused the violation


def normalize(word: str) -> str:
    """Normalize a word for comparison."""
    return word.upper().strip()


def get_word_stem(word: str) -> str:
    """Get a simple stem of a word by removing common suffixes.
    
    This is a simplified stemmer - not as sophisticated as Porter/Snowball
    but sufficient for Codenames variant detection.
    """
    word = word.lower()
    
    # Try removing common suffixes (longest first)
    all_suffixes = sorted(
        set(VERB_SUFFIXES + NOUN_SUFFIXES + ADJ_SUFFIXES),
        key=len,
        reverse=True
    )
    
    for suffix in all_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    
    return word


def are_grammatical_variants(word1: str, word2: str) -> bool:
    """Check if two words are grammatical variants of each other.
    
    This detects:
    - Plurals: CAT → CATS
    - Verb tenses: RUN → RUNNING, RUNS, RAN
    - Comparatives: FAST → FASTER, FASTEST
    - Common derivations: OPERATE → OPERATING, OPERATION
    
    Returns True if they appear to be variants of the same base word.
    """
    w1 = word1.lower().strip()
    w2 = word2.lower().strip()
    
    # Exact match (shouldn't happen, but check anyway)
    if w1 == w2:
        return True
    
    # One is a prefix of the other (like "run" and "running")
    if w1.startswith(w2) or w2.startswith(w1):
        shorter, longer = (w1, w2) if len(w1) < len(w2) else (w2, w1)
        suffix = longer[len(shorter):]
        
        # Check if the suffix is a common grammatical suffix
        if suffix in ['s', 'es', 'ed', 'ing', 'er', 'ers', 'est', 'ly', 'ment', 'tion', 'sion', 'ness']:
            return True
    
    # Check if they share the same stem
    stem1 = get_word_stem(w1)
    stem2 = get_word_stem(w2)
    
    # Stems must be reasonably long and similar
    if len(stem1) >= 3 and len(stem2) >= 3:
        # Check if stems match or one is prefix of other
        if stem1 == stem2:
            return True
        if stem1.startswith(stem2) or stem2.startswith(stem1):
            shorter_stem = stem1 if len(stem1) < len(stem2) else stem2
            if len(shorter_stem) >= 3:
                return True
    
    # Special cases for irregular forms
    irregular_groups = [
        {'run', 'ran', 'running', 'runs', 'runner', 'runners'},
        {'go', 'went', 'going', 'goes', 'gone'},
        {'be', 'is', 'are', 'was', 'were', 'been', 'being'},
        {'have', 'has', 'had', 'having'},
        {'do', 'does', 'did', 'doing', 'done'},
        {'make', 'made', 'making', 'makes', 'maker', 'makers'},
        {'take', 'took', 'taken', 'taking', 'takes', 'taker'},
        {'give', 'gave', 'given', 'giving', 'gives', 'giver'},
        {'see', 'saw', 'seen', 'seeing', 'sees', 'seer'},
        {'know', 'knew', 'known', 'knowing', 'knows'},
        {'think', 'thought', 'thinking', 'thinks', 'thinker'},
        {'get', 'got', 'gotten', 'getting', 'gets'},
        {'say', 'said', 'saying', 'says'},
        {'come', 'came', 'coming', 'comes'},
        {'find', 'found', 'finding', 'finds', 'finder'},
        {'tell', 'told', 'telling', 'tells'},
        {'write', 'wrote', 'written', 'writing', 'writes', 'writer'},
        {'read', 'reading', 'reads', 'reader'},  # 'read' past tense same spelling
        {'stand', 'stood', 'standing', 'stands'},
        {'sit', 'sat', 'sitting', 'sits'},
        {'fall', 'fell', 'fallen', 'falling', 'falls'},
        {'fly', 'flew', 'flown', 'flying', 'flies', 'flyer', 'flier'},
        {'drive', 'drove', 'driven', 'driving', 'drives', 'driver'},
        {'break', 'broke', 'broken', 'breaking', 'breaks', 'breaker'},
        {'speak', 'spoke', 'spoken', 'speaking', 'speaks', 'speaker'},
        {'choose', 'chose', 'chosen', 'choosing', 'chooses'},
        {'begin', 'began', 'begun', 'beginning', 'begins', 'beginner'},
        {'swim', 'swam', 'swum', 'swimming', 'swims', 'swimmer'},
        {'sing', 'sang', 'sung', 'singing', 'sings', 'singer'},
        {'ring', 'rang', 'rung', 'ringing', 'rings', 'ringer'},
        {'drink', 'drank', 'drunk', 'drinking', 'drinks', 'drinker'},
        {'sink', 'sank', 'sunk', 'sinking', 'sinks'},
        {'grow', 'grew', 'grown', 'growing', 'grows', 'grower'},
        {'throw', 'threw', 'thrown', 'throwing', 'throws', 'thrower'},
        {'blow', 'blew', 'blown', 'blowing', 'blows', 'blower'},
        {'draw', 'drew', 'drawn', 'drawing', 'draws', 'drawer'},
        {'hide', 'hid', 'hidden', 'hiding', 'hides', 'hider'},
        {'ride', 'rode', 'ridden', 'riding', 'rides', 'rider'},
        {'bite', 'bit', 'bitten', 'biting', 'bites', 'biter'},
        {'fight', 'fought', 'fighting', 'fights', 'fighter'},
        {'catch', 'caught', 'catching', 'catches', 'catcher'},
        {'teach', 'taught', 'teaching', 'teaches', 'teacher'},
        {'buy', 'bought', 'buying', 'buys', 'buyer'},
        {'bring', 'brought', 'bringing', 'brings', 'bringer'},
        {'build', 'built', 'building', 'builds', 'builder'},
        {'send', 'sent', 'sending', 'sends', 'sender'},
        {'spend', 'spent', 'spending', 'spends', 'spender'},
        {'lend', 'lent', 'lending', 'lends', 'lender'},
        {'lose', 'lost', 'losing', 'loses', 'loser'},
        {'win', 'won', 'winning', 'wins', 'winner'},
        {'keep', 'kept', 'keeping', 'keeps', 'keeper'},
        {'sleep', 'slept', 'sleeping', 'sleeps', 'sleeper'},
        {'feel', 'felt', 'feeling', 'feels', 'feeler'},
        {'leave', 'left', 'leaving', 'leaves', 'leaver'},
        {'mean', 'meant', 'meaning', 'means'},
        {'lead', 'led', 'leading', 'leads', 'leader'},
        {'meet', 'met', 'meeting', 'meets'},
        {'pay', 'paid', 'paying', 'pays', 'payer'},
        {'sell', 'sold', 'selling', 'sells', 'seller'},
        {'hold', 'held', 'holding', 'holds', 'holder'},
        {'hear', 'heard', 'hearing', 'hears', 'hearer'},
        {'cut', 'cutting', 'cuts', 'cutter'},
        {'put', 'putting', 'puts'},
        {'set', 'setting', 'sets', 'setter'},
        {'hit', 'hitting', 'hits', 'hitter'},
        {'shut', 'shutting', 'shuts'},
        {'let', 'letting', 'lets'},
        {'hurt', 'hurting', 'hurts'},
    ]
    
    for group in irregular_groups:
        if w1 in group and w2 in group:
            return True
    
    return False


def is_multiple_words(clue: str) -> Tuple[bool, str]:
    """Check if a clue contains multiple words (invalid unless exception applies).
    
    Exceptions:
    - Hyphenated compounds (mother-in-law)
    - Common proper nouns (would need a dictionary)
    - Acronyms
    
    Returns (is_multiple, reason)
    """
    clue = clue.strip()
    
    # Check for spaces
    if ' ' in clue:
        words = clue.split()
        if len(words) > 1:
            # Check if it looks like a proper noun (capitalized words)
            # This is a heuristic - not perfect
            if all(w[0].isupper() for w in words if w):
                # Likely a proper noun like "New York" - allow it
                return False, ""
            
            return True, f"Multiple words: '{clue}'"
    
    return False, ""


def references_letter_count(clue: str) -> Tuple[bool, str]:
    """Check if clue references letter counts."""
    clue_lower = clue.lower()
    
    for pattern in LETTER_COUNT_PATTERNS:
        if re.search(pattern, clue_lower):
            return True, f"References letter count: '{clue}'"
    
    return False, ""


def references_position(clue: str) -> Tuple[bool, str]:
    """Check if clue references board positions."""
    clue_lower = clue.lower()
    
    for pattern in POSITION_PATTERNS:
        if re.search(pattern, clue_lower):
            return True, f"References board position: '{clue}'"
    
    return False, ""


def validate_clue(
    clue: str,
    board_words: List[str],
) -> ValidationResult:
    """Validate a clue against all Codenames rules.
    
    Rules checked:
    1. No multiple words (with exceptions)
    2. No exact match to board words
    3. No grammatical variants of board words
    4. No references to letter count
    5. No references to board position
    
    Args:
        clue: The proposed clue word
        board_words: All words currently on the board
        
    Returns:
        ValidationResult with is_valid, reason, and details
    """
    clue = clue.strip()
    
    # Rule 1: Multiple words
    is_multi, reason = is_multiple_words(clue)
    if is_multi:
        return ValidationResult(
            is_valid=False,
            reason=reason,
            rule_violated="multiple_words",
        )
    
    # Normalize for comparison
    clue_upper = normalize(clue)
    board_upper = {normalize(w): w for w in board_words}
    
    # Rule 2: Exact match
    if clue_upper in board_upper:
        original_word = board_upper[clue_upper]
        return ValidationResult(
            is_valid=False,
            reason=f"Exact match to board word: {original_word}",
            rule_violated="exact_match",
            related_word=original_word,
        )
    
    # Rule 3: Grammatical variant
    for board_word_upper, original_word in board_upper.items():
        if are_grammatical_variants(clue_upper, board_word_upper):
            return ValidationResult(
                is_valid=False,
                reason=f"Grammatical variant of board word: {original_word}",
                rule_violated="grammatical_variant",
                related_word=original_word,
            )
    
    # Rule 4: Letter count reference
    is_letter_ref, reason = references_letter_count(clue)
    if is_letter_ref:
        return ValidationResult(
            is_valid=False,
            reason=reason,
            rule_violated="letter_count_reference",
        )
    
    # Rule 5: Position reference
    is_pos_ref, reason = references_position(clue)
    if is_pos_ref:
        return ValidationResult(
            is_valid=False,
            reason=reason,
            rule_violated="position_reference",
        )
    
    # All checks passed
    return ValidationResult(
        is_valid=True,
        reason="Valid clue",
    )


class RuleBasedReferee:
    """Deterministic rule-based referee for Codenames.
    
    This class provides clue validation without LLM calls,
    making it suitable for use in DSPy optimization loops.
    """
    
    # Penalty score for invalid clues
    # An invalid clue ends your turn and reveals one opponent agent
    # This is roughly equivalent to guessing an enemy agent (-1.0)
    INVALID_CLUE_PENALTY = -1.0
    
    def validate(
        self,
        clue: str,
        board_words: List[str],
    ) -> ValidationResult:
        """Validate a clue against the board.
        
        Args:
            clue: The proposed clue
            board_words: All words on the board
            
        Returns:
            ValidationResult
        """
        return validate_clue(clue, board_words)
    
    def get_penalty(self, result: ValidationResult) -> float:
        """Get the penalty score for an invalid clue.
        
        Returns 0 if valid, INVALID_CLUE_PENALTY if invalid.
        """
        if result.is_valid:
            return 0.0
        return self.INVALID_CLUE_PENALTY


# Module-level instance for convenience
referee = RuleBasedReferee()


def validate_clue_quick(clue: str, board_words: List[str]) -> bool:
    """Quick validation check - returns True if valid, False if invalid."""
    return validate_clue(clue, board_words).is_valid


if __name__ == "__main__":
    # Quick test
    test_board = [
        "CAT", "DOG", "RUNNING", "HOUSE", "TREE",
        "WATER", "FIRE", "EARTH", "AIR", "SPACE",
        "BOOK", "PEN", "PAPER", "TABLE", "CHAIR",
        "RED", "BLUE", "GREEN", "YELLOW", "BLACK",
        "ONE", "TWO", "THREE", "FOUR", "FIVE",
    ]
    
    test_cases = [
        ("ANIMAL", True, "Valid - semantic connection"),
        ("CAT", False, "Invalid - exact match"),
        ("CATS", False, "Invalid - grammatical variant"),
        ("RUNS", False, "Invalid - grammatical variant of RUNNING"),
        ("RUNNER", False, "Invalid - grammatical variant of RUNNING"),
        ("FOREST", True, "Valid - different word"),
        ("Five letters", False, "Invalid - references letter count"),
        ("Top row", False, "Invalid - references position"),
        ("New York", True, "Valid - proper noun exception"),
        ("CIA", True, "Valid - acronym"),
        ("the cat sat", False, "Invalid - multiple words"),
    ]
    
    print("Rule-Based Referee Test Cases")
    print("=" * 60)
    
    for clue, expected_valid, description in test_cases:
        result = validate_clue(clue, test_board)
        status = "✅" if result.is_valid == expected_valid else "❌"
        valid_str = "VALID" if result.is_valid else "INVALID"
        
        print(f"{status} Clue: '{clue}' → {valid_str}")
        print(f"   Expected: {'VALID' if expected_valid else 'INVALID'} ({description})")
        if not result.is_valid:
            print(f"   Reason: {result.reason}")
        print()

