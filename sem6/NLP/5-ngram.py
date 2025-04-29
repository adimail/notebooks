import re
import random
from collections import defaultdict


class NGramModel:
    def __init__(self, n):
        """Initializes the N-gram model."""
        if n < 2:
            raise ValueError("N must be at least 2 for N-grams.")
        self.n = n
        # Stores mapping: (n-1)-gram tuple -> list of possible next words
        self.ngrams = defaultdict(list)
        # Stores counts for probability calculation (optional, not used in this generation)
        # self.ngram_counts = defaultdict(int)
        # self.context_counts = defaultdict(int)

    def preprocess(self, text):
        """Cleans and tokenizes the text."""
        text = text.lower()
        # Keep letters, numbers, and whitespace. Remove other punctuation.
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = text.split()
        return tokens

    def train(self, text):
        """Trains the N-gram model on the provided text."""
        tokens = self.preprocess(text)

        # Ensure text is long enough to form at least one N-gram
        if len(tokens) < self.n:
            print(
                f"Warning: Text is too short ({len(tokens)} words) to train a {self.n}-gram model."
            )
            return

        # Iterate through the tokens to build N-grams
        for i in range(len(tokens) - self.n + 1):
            # The context is the first n-1 words of the N-gram
            context = tuple(tokens[i : i + self.n - 1])
            # The target word is the nth word
            target_word = tokens[i + self.n - 1]

            self.ngrams[context].append(target_word)
            # Optional: Update counts if calculating probabilities
            # self.ngram_counts[context + (target_word,)] += 1
            # self.context_counts[context] += 1

    def generate_text(self, seed, num_words=20):
        """Generates text starting with a seed sequence."""
        seed_tokens = self.preprocess(seed)

        # N-gram model needs n-1 words as context to predict the next
        required_seed_length = self.n - 1
        if len(seed_tokens) < required_seed_length:
            raise ValueError(
                f"Seed text must be at least {required_seed_length} words long for an N={self.n} model."
            )

        # Start with the last n-1 words of the seed as the initial context
        current_context_list = seed_tokens[-required_seed_length:]
        output = list(current_context_list)  # Start output with the context

        for _ in range(num_words):
            current_context_tuple = tuple(current_context_list)

            # Get possible next words for the current context
            possible_next_words = self.ngrams.get(current_context_tuple)

            if not possible_next_words:
                # If context not seen during training, stop generation
                # print(f"\nWarning: Context {current_context_tuple} not found. Stopping generation.")
                break

            # Choose the next word randomly from the possibilities
            next_word = random.choice(possible_next_words)
            output.append(next_word)

            # Update the context: slide the window one word forward
            current_context_list = current_context_list[1:] + [next_word]

        return " ".join(output)


# --- Example Usage from Description ---
text_corpus = """
Deep learning models like CNN, RNN, and transformers are powerful
tools for solving AI problems.
They learn patterns from data and generalize to unseen examples.
AI models learn fast. Deep learning is powerful.
"""

# Create and train a bigram model (n=2)
# A bigram model uses the previous 1 word (n-1) to predict the next.
bigram_model = NGramModel(n=2)
print("Training Bigram (N=2) model...")
bigram_model.train(text_corpus)
print("Training complete.")

# Create and train a trigram model (n=3)
# A trigram model uses the previous 2 words (n-1) to predict the next.
trigram_model = NGramModel(n=3)
print("\nTraining Trigram (N=3) model...")
trigram_model.train(text_corpus)
print("Training complete.")


# Generate text using the bigram model
seed_bigram = "deep learning"  # Needs n-1 = 1 word
print(f"\nGenerating text with Bigram (N=2), Seed: '{seed_bigram}'")
generated_bigram = bigram_model.generate_text(seed_bigram, num_words=15)
# Note: The original output example seems to start generation *after* the seed.
# This implementation includes the seed context in the output.
# To match the example exactly, adjust slicing or printing.
print("Generated (Bigram):", generated_bigram)


# Generate text using the trigram model
seed_trigram = "deep learning"  # Needs n-1 = 2 words
print(f"\nGenerating text with Trigram (N=3), Seed: '{seed_trigram}'")
generated_trigram = trigram_model.generate_text(seed_trigram, num_words=15)
print("Generated (Trigram):", generated_trigram)

# Example matching the output format provided in the image (starting after seed)
print("\n--- Matching Provided Output Format ---")
seed_provided = "deep learning"
model_provided = NGramModel(n=2)  # As per example output structure
model_provided.train(text_corpus)
# Generate, but slice off the seed part for printing
full_generated = model_provided.generate_text(seed_provided, num_words=15)
# Find where the seed ends in the output and print the rest
seed_tokens_len = len(model_provided.preprocess(seed_provided))
output_tokens = full_generated.split()
print_output = " ".join(
    output_tokens[seed_tokens_len - 1 :]
)  # Adjust index slightly based on how generate works
print(f"Seed: '{seed_provided}'")
# The provided output seems to be from a slightly different generation logic or corpus
# This code replicates the class structure but generation might differ slightly.
# Let's try to generate more words and see if the pattern emerges
longer_generated = model_provided.generate_text(seed_provided, num_words=30)
print("Generated (attempting to match example):", longer_generated)
