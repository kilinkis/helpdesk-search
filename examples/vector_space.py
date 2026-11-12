"""Distance and similarity in vector space: cosine, dot product and Euclidean.

Starts with 2-D vectors you can picture, then does the same with real
384-dimensional sentence embeddings.
"""

import numpy as np

from helpdesk_search.embeddings import embed


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


# 2-D: b points the same way as a but is twice as long, c points elsewhere.
a, b, c = np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([1.0, -1.0])
print("2-D vectors")
print(f"  a={a}, b={b}, c={c}")
print(f"  cosine(a, b)    = {cosine(a, b):.2f}   same direction, length ignored")
print(f"  cosine(a, c)    = {cosine(a, c):.2f}   perpendicular")
print(f"  euclidean(a, b) = {euclidean(a, b):.2f}   length matters")
print(f"  dot(a, b)       = {a @ b:.2f}   grows with length")

# Sentence embeddings: hundreds of dimensions, normalized to length 1.
sentences = [
    "I was charged twice this month",
    "There is a duplicate payment on my card",
    "How do I reset my password?",
]
vectors = embed(sentences)
print(f"\nSentence embeddings: {vectors.shape[1]} dimensions, length {np.linalg.norm(vectors[0]):.2f}")
print(f"  first numbers of '{sentences[0]}': {np.round(vectors[0][:5], 3)} ...")
for i, j in [(0, 1), (0, 2)]:
    print(f"\n  '{sentences[i]}' vs '{sentences[j]}'")
    print(f"    cosine {cosine(vectors[i], vectors[j]):.3f}  dot {vectors[i] @ vectors[j]:.3f}  euclidean {euclidean(vectors[i], vectors[j]):.3f}")

# With unit-length vectors the three measures rank results the same way:
# cosine == dot product, and euclidean distance = sqrt(2 - 2 * cosine).
print("\nFor unit vectors, cosine equals the dot product, and Euclidean distance = sqrt(2 - 2 * cosine).")
