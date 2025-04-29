# experiment_4.py
import nltk
from nltk.corpus import wordnet as wn

# --- Download WordNet data (run once) ---
# try:
#     wn.ensure_loaded()
# except LookupError:
#     print("Downloading WordNet...")
#     nltk.download('wordnet')
# ---


def explore_word_semantics(word):
    """
    Explores semantic relationships for a word using WordNet.
    """
    print(f"\n--- Exploring Semantics for '{word}' ---")
    synsets = wn.synsets(word)

    if not synsets:
        print("Word not found in WordNet.")
        return

    # Use the first synset (most common sense) for detailed exploration
    main_synset = synsets[0]
    print(f"Most Common Synset: {main_synset.name()}")
    print(f"Definition: {main_synset.definition()}")
    print(f"Examples: {main_synset.examples()}")

    # 1. Synonyms (lemmas in the same synset)
    synonyms = set()
    for syn in synsets:  # Check all synsets for the word
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.discard(word)  # Remove the original word itself
    print(f"\nSynonyms (across senses): {list(synonyms)}")

    # 2. Antonyms (associated with specific lemmas)
    antonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            for antonym_lemma in lemma.antonyms():
                antonyms.add(antonym_lemma.name())
    if antonyms:
        print(f"Antonyms: {list(antonyms)}")
    else:
        print("Antonyms: None found")

    # 3. Hyponyms (more specific concepts) - using main synset
    hyponyms = main_synset.hyponyms()
    if hyponyms:
        print(f"\nHyponyms (specific types of '{main_synset.name()}'):")
        for hypo in hyponyms[:5]:  # Show first 5
            print(f"- {hypo.name()} ({hypo.definition()})")
    else:
        print("\nHyponyms: None found for the main sense.")

    # 4. Hypernyms (broader concepts) - using main synset
    hypernyms = main_synset.hypernyms()
    if hypernyms:
        print(f"\nHypernyms (broader categories for '{main_synset.name()}'):")
        for hyper in hypernyms:
            print(f"- {hyper.name()} ({hyper.definition()})")
        # Root hypernym
        root_hyper = main_synset.root_hypernyms()
        if root_hyper:
            print(f"Root Hypernym: {root_hyper[0].name()}")
    else:
        print("\nHypernyms: None found for the main sense.")

    # 5. Meronyms (part-of relationships) - using main synset
    part_meronyms = main_synset.part_meronyms()
    substance_meronyms = main_synset.substance_meronyms()
    if part_meronyms or substance_meronyms:
        print(f"\nMeronyms (parts of '{main_synset.name()}'):")
        for mero in part_meronyms[:5]:
            print(f"- Part: {mero.name()}")
        for mero in substance_meronyms[:5]:
            print(f"- Substance: {mero.name()}")
    else:
        print("\nMeronyms: None found for the main sense.")

    # 6. Holonyms (whole-of relationships) - using main synset
    member_holonyms = main_synset.member_holonyms()
    part_holonyms = main_synset.part_holonyms()
    substance_holonyms = main_synset.substance_holonyms()
    if member_holonyms or part_holonyms or substance_holonyms:
        print(f"\nHolonyms ('{main_synset.name()}' is part of):")
        for holo in member_holonyms[:5]:
            print(f"- Member Of: {holo.name()}")
        for holo in part_holonyms[:5]:
            print(f"- Part Of: {holo.name()}")
        for holo in substance_holonyms[:5]:
            print(f"- Substance Of: {holo.name()}")
    else:
        print("\nHolonyms: None found for the main sense.")


# --- Example Usage from Description ---
sentence_words = ["dog", "cat", "park", "chased"]  # Added 'chased' for variety

for w in sentence_words:
    explore_word_semantics(w)
    print("=" * 40)
