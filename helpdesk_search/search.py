"""Index the help center and search it.

Indexing:  record text -> embed -> upsert (id, vector, metadata) into a store
Searching: question    -> embed -> query the store -> closest records first
"""

import os

import pandas as pd

from helpdesk_search.data import load_articles, load_sections
from helpdesk_search.embeddings import DEFAULT_MODEL, dimension, embed
from helpdesk_search.store import Match, VectorStore, get_store

LEVELS = ("sections", "articles")


def index_records(store: VectorStore, records: pd.DataFrame, model: str = DEFAULT_MODEL) -> int:
    """Embed every record's text and upsert it with its id and metadata."""
    vectors = embed(records["text"].tolist(), model)
    store.upsert(zip(records["unique_id"], vectors, records["metadata"]))
    store.wait_until(lambda: store.count() >= len(records))
    return len(records)


def open_index(level: str = "sections", model: str = DEFAULT_MODEL, backend: str | None = None) -> VectorStore:
    """A store for one level and model, filled with the help center if it's empty.

    With Pinecone there is one index per model, because an index has a fixed
    dimension, and one namespace per level inside it.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    name = f"{os.getenv('PINECONE_INDEX', 'helpdesk-search')}-{model}"
    store = get_store(dimension(model), name=name, namespace=level, backend=backend)
    if store.count() == 0:
        index_records(store, load_sections() if level == "sections" else load_articles(), model)
    return store


def search(
    store: VectorStore, question: str, model: str = DEFAULT_MODEL, top_k: int = 3, filter: dict | None = None
) -> list[Match]:
    """The records closest in meaning to the question. Use the same model the store was indexed with."""
    return store.query(embed([question], model)[0], top_k=top_k, filter=filter)


def related(store: VectorStore, id: str, top_k: int = 3, filter: dict | None = None) -> list[Match]:
    """Records similar to a stored one, for "related articles". Queries by id, so nothing is embedded."""
    matches = store.query(id=id, top_k=top_k + 1, filter=filter)
    return [m for m in matches if m.id != id][:top_k]


def describe(match: Match) -> str:
    """A one-line label for a match from either level."""
    title = match.metadata.get("article_title", "")
    if "section_title" in match.metadata:
        title += f" > {match.metadata['section_title']}"
    return f"{match.score:.3f}  [{match.id:>4}]  {title}"
