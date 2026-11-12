"""An embedding model turns text into a vector so that similar meanings land close together."""

from helpdesk_search.embeddings import MODELS, dimension, embed, get_encoder

model = "minilm"
print(f"Model: {MODELS[model]} ({dimension(model)} dimensions, max {get_encoder(model).max_seq_length} tokens)")

sentences = [
    "I was charged twice",
    "Why is there a duplicate payment?",
    "I forgot my password",
    "I can't sign in to my account",
    "The app crashes when I open it",
]
vectors = embed(sentences, model)
print(f"embed() returned an array of shape {vectors.shape}: one row per sentence.\n")

# Similarity of every pair. Rows are unit length, so a matrix product gives cosine similarity.
similarity = vectors @ vectors.T
width = max(len(s) for s in sentences)
print(" " * (width + 2) + "  ".join(f"{i:>5}" for i in range(len(sentences))))
for i, sentence in enumerate(sentences):
    print(f"{sentence:<{width}}  " + "  ".join(f"{score:5.2f}" for score in similarity[i]) + f"   {i}")

print("\nThe paraphrases score highest with each other, even with no words in common.")
