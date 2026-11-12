"""Turn text into vectors with sentence-transformers models that run locally.

No API key is needed. Each model is downloaded from Hugging Face the first time
it is used and cached on disk after that.
"""

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer  # noqa: E402
from huggingface_hub.utils import logging as hub_logging  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

transformers_logging.disable_progress_bar()
hub_logging.set_verbosity_error()  # hides the "unauthenticated requests" notice; downloads work without a token

MODELS = {
    # Small and fast: 6 layers, 384 dimensions, ~90 MB.
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    # A BERT model (DistilBERT) trained for search: 768 dimensions, ~265 MB.
    "bert": "sentence-transformers/multi-qa-distilbert-cos-v1",
}
DEFAULT_MODEL = "minilm"


@lru_cache
def get_encoder(model: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load a model once per process. Accepts a short name from MODELS or any Hugging Face id."""
    return SentenceTransformer(MODELS.get(model, model))


def embed(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """One row per text. Rows are normalized to length 1, so cosine similarity equals the dot product."""
    return get_encoder(model).encode(texts, normalize_embeddings=True, show_progress_bar=False)


def dimension(model: str = DEFAULT_MODEL) -> int:
    """How many numbers each vector has. A vector index must be created with this size."""
    return get_encoder(model).get_embedding_dimension()
