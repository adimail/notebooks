import nltk

# --- Download necessary NLTK data (run once) ---
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     print("Downloading 'punkt' tokenizer...")
#     nltk.download('punkt')
# try:
#     nltk.data.find('taggers/averaged_perceptron_tagger')
# except nltk.downloader.DownloadError:
#     print("Downloading 'averaged_perceptron_tagger'...")
#     nltk.download('averaged_perceptron_tagger')
# ---


def pos_tag_sentence(sentence):
    """
    Tokenizes a sentence and performs Part-of-Speech tagging using NLTK.
    """
    # Tokenize the sentence into words
    tokens = nltk.word_tokenize(sentence)

    # Perform POS tagging
    tagged_tokens = nltk.pos_tag(tokens)

    return tagged_tokens


# --- Example Usage from Description ---
sentence = "The quick brown fox jumps over the lazy dog."

print(f"Sentence: {sentence}")
tagged_sentence = pos_tag_sentence(sentence)
print("\nPOS Tagging Results:")
for word, tag in tagged_sentence:
    print(f"- '{word}' -> {tag}")

# Example of NLTK tag explanations (optional)
# nltk.download('tagsets') # Uncomment to download tagset help
# print("\nExplanation for VBZ tag:")
# nltk.help.upenn_tagset('VBZ')
# print("\nExplanation for NN tag:")
# nltk.help.upenn_tagset('NN')
