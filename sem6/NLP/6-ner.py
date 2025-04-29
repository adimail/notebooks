import spacy
from spacy import displacy
from sklearn.metrics import classification_report

# pip install spacy scikit-learn and python -m spacy download en_core_web_sm

# Load spaCy's small English model
# Make sure you have run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Sample texts from description
texts = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Barack Obama was born in Hawaii.",
    "Elon Musk is the CEO of SpaceX and Tesla.",
    # Add more diverse examples if needed
]

print("--- Named Entity Recognition ---")
# Extract entities and their labels
all_predicted_labels = []
all_texts_entities = []

for text in texts:
    doc = nlp(text)
    print(f"\nText: {text}")
    print("Entities:")
    entities_in_text = []
    for ent in doc.ents:
        print(f"{ent.text:<20} -> {ent.label_}")
        entities_in_text.append((ent.text, ent.label_))
    all_texts_entities.append(entities_in_text)
    # Optionally visualize it in browser (uncomment to run)
    # displacy.serve(doc, style="ent") # This will start a local server

# --- Evaluation Example (using the first sentence's entities) ---
print("\n--- Evaluation Example ---")
# Example ground truth labels for the first sentence:
# "Apple": ORG, "U.K.": GPE, "$1 billion": MONEY
y_true_example = ["ORG", "GPE", "MONEY"]

# Get predicted labels from spaCy for the first sentence
if all_texts_entities:
    first_sentence_entities = all_texts_entities[0]
    y_pred_example = [label for text, label in first_sentence_entities]
else:
    y_pred_example = []


print(f"Sample Text: '{texts[0]}'")
print(f"Ground Truth Labels : {y_true_example}")
print(f"Predicted Labels    : {y_pred_example}")

# Important Note on Evaluation:
# Comparing lists directly like this only works if the number and order of
# predicted entities exactly match the ground truth. A proper NER evaluation
# uses metrics like precision, recall, F1-score based on matching entity spans
# and types (e.g., using libraries like 'seqeval').
# sklearn.metrics.classification_report is typically for classification tasks
# where each item has one label, not sequence labeling like NER.
# However, we replicate the *usage* shown in the image.

# Replicating the classification_report usage from the image:
# This assumes a scenario where you have parallel lists of true/pred labels
# for *individual entities* across a dataset, not just one sentence.
# The example in the image seems to use labels from the first sentence
# as if they were independent classifications.

if len(y_true_example) == len(y_pred_example) and y_true_example:
    print("\nClassification Report (Illustrative - based on image example):")
    # Get unique labels present in either true or predicted
    labels = sorted(list(set(y_true_example + y_pred_example)))
    print(classification_report(y_true_example, y_pred_example, labels=labels))
else:
    print("\nCould not generate classification report:")
    print("Number of true and predicted labels differ, or no entities found.")
    if all_texts_entities:
        print(f"Predicted: {y_pred_example}")
        print(f"True: {y_true_example}")

# --- Output matching the image's classification report ---
# The image shows a perfect score (1.00) for ORG, GPE, MONEY.
# This implies the y_true and y_pred used for that specific report were identical.
print("\n--- Replicating Image's Classification Report Data ---")
y_true_img = ["ORG", "GPE", "MONEY"]
y_pred_img = [
    "ORG",
    "GPE",
    "MONEY",
]  # Assuming prediction matched truth for the report shown
labels_img = sorted(list(set(y_true_img + y_pred_img)))
print("Assuming y_true = ['ORG', 'GPE', 'MONEY'] and y_pred = ['ORG', 'GPE', 'MONEY']")
print(classification_report(y_true_img, y_pred_img, labels=labels_img))
