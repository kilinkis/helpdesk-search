"""Compare two embedding models on the same questions: MiniLM and a BERT model.

Each model needs its own index, because vectors from different models have
different dimensions and aren't comparable.
"""

import time

from helpdesk_search.embeddings import MODELS, dimension, embed
from helpdesk_search.search import describe, open_index, search

QUESTIONS = [
    "I was charged twice",
    "my changes didn't show up on my laptop",
    "add a coworker to my workspace",
    "the app keeps closing",
]
texts = [f"Sample help center sentence number {i} about billing and refunds." for i in range(500)]

for model in MODELS:
    embed(texts, model)  # warm up: the first run loads the model and is slower
    start = time.perf_counter()
    embed(texts, model)
    elapsed = time.perf_counter() - start
    print(f"{model:<6} {MODELS[model]}")
    print(f"       {dimension(model)} dimensions, {len(texts)} sentences embedded in {elapsed:.2f}s")

stores = {model: open_index("sections", model) for model in MODELS}
for question in QUESTIONS:
    print(f"\n{question}")
    for model, store in stores.items():
        print(f"  {model:<6} " + describe(search(store, question, model, top_k=1)[0]))

print("\nScores from different models aren't on the same scale, so compare the rankings, not the numbers.")
