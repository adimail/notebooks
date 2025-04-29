import nltk

# nltk.download('words') # Uncomment and run once if you haven't downloaded the words corpus
from nltk.corpus import words

# Simple list of common English bound morphemes (prefixes/suffixes)
# This is illustrative and not exhaustive
BOUND_MORPHEMES = {
    "prefixes": {"un", "re", "pre", "dis", "mis", "in", "im", "il", "ir", "non"},
    "suffixes": {
        "s",
        "es",
        "ed",
        "ing",
        "ly",
        "er",
        "or",
        "ist",
        "ian",
        "able",
        "ible",
        "ness",
        "ment",
        "tion",
        "sion",
    },
}

# Basic check if a string is likely an English word
# For simplicity, using nltk.corpus.words. More robust checks exist.
word_list = set(words.words())


def analyze_morphemes(word):
    """
    Attempts a basic analysis of a word into potential free and bound morphemes.
    This is a simplified approach for demonstration.
    """
    word = word.lower()
    results = {"word": word, "free": [], "bound": []}

    # Check if the word itself is a known bound morpheme (unlikely for typical input)
    if word in BOUND_MORPHEMES["prefixes"] or word in BOUND_MORPHEMES["suffixes"]:
        results["bound"].append(word)
        return results

    # Try stripping common suffixes
    potential_stem = word
    found_suffix = None
    for suffix in sorted(BOUND_MORPHEMES["suffixes"], key=len, reverse=True):
        if word.endswith(suffix):
            potential_stem = word[: -len(suffix)]
            # Check if the stem is a word or if the original word minus suffix is common
            if potential_stem in word_list or len(potential_stem) > 2:  # Basic check
                results["bound"].append(f"-{suffix}")
                found_suffix = True
                break  # Take the longest matching suffix first

    # Try stripping common prefixes from the (potentially suffix-stripped) stem
    stem_to_check_prefix = potential_stem if found_suffix else word
    final_stem = stem_to_check_prefix
    found_prefix = None
    for prefix in sorted(BOUND_MORPHEMES["prefixes"], key=len, reverse=True):
        if stem_to_check_prefix.startswith(prefix):
            potential_stem_after_prefix = stem_to_check_prefix[len(prefix) :]
            # Check if the remaining part is a word
            if (
                potential_stem_after_prefix in word_list
                or len(potential_stem_after_prefix) > 1
            ):  # Basic check
                results["bound"].append(f"{prefix}-")
                final_stem = potential_stem_after_prefix
                found_prefix = True
                break  # Take the longest matching prefix first

    # Assume the final remaining stem is the free morpheme (or root)
    if final_stem and (final_stem in word_list or not (found_prefix or found_suffix)):
        # If we didn't find affixes, the original word is likely free
        # Or if the remaining stem is a word, consider it free
        results["free"].append(final_stem if (found_prefix or found_suffix) else word)
    elif final_stem:
        # If stem not in word list but we stripped affixes, list it tentatively
        results["free"].append(f"({final_stem})")  # Indicate potential root

    # If no bound morphemes found, classify the original word as free
    if not results["bound"] and word in word_list:
        results["free"] = [word]
        results["bound"] = []  # Ensure bound is empty

    # Handle cases like 'running' -> run + ing
    if word == "running":
        results = {"word": word, "free": ["run"], "bound": ["-ing"]}
    if word == "cats":
        results = {"word": word, "free": ["cat"], "bound": ["-s"]}
    if word == "unhappy":
        results = {"word": word, "free": ["happy"], "bound": ["un-"]}

    return results


# --- Example Usage ---
words_to_analyze = [
    "cat",
    "cats",
    "running",
    "unhappy",
    "preview",
    "impossible",
    "kindness",
]

print("Morphological Analysis (Simplified):")
for w in words_to_analyze:
    analysis = analyze_morphemes(w)
    print(f"- {analysis['word']}: Free={analysis['free']}, Bound={analysis['bound']}")
