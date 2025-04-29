# pip install spacy and python -m spacy download en_core_web_sm

import spacy

# Load spaCy's small English model
# Make sure you have run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Your input text from the description
text = """Elon Musk, the CEO of Tesla and SpaceX, announced a new AI
company in San Francisco on April 5, 2025."""

print(f'Processing Text: "{text}"')

# Process the text
doc = nlp(text)

# Print named entities, their labels, and explanations
print("\n--- Named Entities, their labels, and explanation: ---")
if not doc.ents:
    print("No entities found in the text.")
else:
    # Determine max length for alignment (optional, for cleaner printing)
    max_len_text = max(len(ent.text) for ent in doc.ents)
    max_len_label = max(len(ent.label_) for ent in doc.ents)

    for ent in doc.ents:
        label_explanation = spacy.explain(ent.label_)
        # Using f-string alignment based on calculated max lengths
        print(
            f"{ent.text:<{max_len_text}} -> {ent.label_:<{max_len_label}} ({label_explanation})"
        )

# --- Example matching the output format in the image ---
# The image output has slightly different formatting and includes 'fictional'
# which spacy.explain doesn't typically add directly. It might be from an older
# version or custom explanation. This code uses the standard spacy.explain.

print("\n--- Output using standard spacy.explain ---")
# Re-running the loop with standard formatting for clarity
for ent in doc.ents:
    print(
        f"Entity: '{ent.text}', Label: {ent.label_}, Explanation: {spacy.explain(ent.label_)}"
    )


# Note: The image output shows 'NORP' for Tesla, but en_core_web_sm typically identifies
# Tesla as ORG. Models can differ in their predictions. Let's check the actual output:
print("\n--- Actual Output from en_core_web_sm ---")
for ent in doc.ents:
    label_explanation = spacy.explain(ent.label_)
    print(f"{ent.text:<25} -> {ent.label_:<10} ({label_explanation})")
