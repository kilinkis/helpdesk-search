import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from helpdesk_search.data import load_articles, load_help_center, load_sections
from helpdesk_search.embeddings import dimension, embed
from helpdesk_search.search import index_records, open_index, related, search
from helpdesk_search.store import InMemoryStore, PineconeStore, get_store, matches_filter

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


@pytest.fixture(autouse=True)
def memory_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "memory")


@pytest.fixture(scope="module")
def sections():
    return open_index("sections", backend="memory")


@pytest.fixture(scope="module")
def articles():
    return open_index("articles", backend="memory")


# Data preparation


def test_section_records_have_unique_ids_and_context():
    records = load_sections()
    assert len(records) == len(load_help_center())
    assert records["unique_id"].is_unique
    first = records.iloc[0]
    assert first["unique_id"] == "1-1"
    assert first["text"].startswith("Understanding your bill. When you are charged.")
    assert first["metadata"]["category"] == "billing"


def test_article_records_include_every_section():
    records = load_articles()
    assert records["unique_id"].tolist() == [str(i) for i in range(1, 11)]
    assert "Charged twice:" in records.iloc[1]["text"] and "Requesting a refund:" in records.iloc[1]["text"]


# Embeddings


def test_embeddings_are_unit_length_and_capture_meaning():
    vectors = embed(["I was charged twice", "There is a duplicate payment", "I forgot my password"])
    assert vectors.shape == (3, dimension())
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


# In-memory store


def test_upsert_overwrites_and_query_ranks_by_cosine():
    store = InMemoryStore(2)
    store.upsert([("a", [1, 0], {}), ("b", [0, 1], {}), ("a", [1, 1], {"v": 2})])
    assert store.count() == 2
    matches = store.query([1, 1], top_k=2)
    assert [m.id for m in matches] == ["a", "b"]
    assert matches[0].score == pytest.approx(1.0) and matches[0].metadata == {"v": 2}


def test_dotproduct_rewards_length():
    store = InMemoryStore(2, metric="dotproduct")
    store.upsert([("short", [1, 0], {}), ("long", [3, 1], {})])
    assert store.query([1, 0], top_k=1)[0].id == "long"


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError, match="2 numbers"):
        InMemoryStore(2).upsert([("a", [1, 0, 0], {})])


def test_update_merges_metadata_and_replaces_vector():
    store = InMemoryStore(2)
    store.upsert([("a", [1, 0], {"title": "old", "minutes": 3})])
    store.update("a", vector=[0, 1], metadata={"title": "new"})
    store.update("missing", metadata={"title": "ignored"})
    vector, metadata = store.fetch(["a"])["a"]
    assert vector == [0, 1] and metadata == {"title": "new", "minutes": 3}
    assert store.fetch(["missing"]) == {}


def test_delete_and_query_by_id():
    store = InMemoryStore(2)
    store.upsert([("a", [1, 0], {}), ("b", [0.9, 0.1], {}), ("c", [0, 1], {})])
    store.delete(["c", "not-there"])
    assert store.count() == 2
    assert [m.id for m in store.query(id="a", top_k=2)] == ["a", "b"]


@pytest.mark.parametrize(
    ("filter", "expected"),
    [
        ({"category": "billing"}, True),
        ({"category": {"$ne": "billing"}}, False),
        ({"category": {"$in": ["account", "billing"]}}, True),
        ({"minutes": {"$gte": 3, "$lt": 5}}, True),
        ({"minutes": {"$gt": 3}}, False),
        ({"tags": "refund"}, True),
        ({"tags": {"$nin": ["refund"]}}, False),
        ({"missing": {"$exists": False}}, True),
        ({"$or": [{"category": "account"}, {"minutes": 3}]}, True),
        ({"$and": [{"category": "billing"}, {"minutes": 4}]}, False),
    ],
)
def test_metadata_filters(filter, expected):
    metadata = {"category": "billing", "minutes": 3, "tags": ["refund", "card"]}
    assert matches_filter(metadata, filter) is expected


def test_get_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="VECTOR_STORE"):
        get_store(2, backend="redis")


# Search over the help center


@pytest.mark.parametrize(
    ("question", "expected_id"),
    [
        ("I was charged twice", "2-1"),
        ("How long does the reset link last?", "4-1"),
        ("my changes didn't show up on my laptop", "8-2"),
        ("Can I get a copy of my personal data?", "10-2"),
    ],
)
def test_section_search_finds_the_answer(sections, question, expected_id):
    assert search(sections, question, top_k=1)[0].id == expected_id


def test_sections_are_more_precise_than_articles(articles, sections):
    question = "How long does the reset link last?"
    assert search(articles, question, top_k=1)[0].id != "4"
    assert search(sections, question, top_k=1)[0].metadata["article_id"] == "4"


def test_filter_restricts_results(sections):
    matches = search(sections, "how do I delete something", top_k=5, filter={"category": "billing"})
    assert matches and {m.metadata["category"] for m in matches} == {"billing"}


def test_related_excludes_the_current_article(articles):
    matches = related(articles, "1", top_k=2)
    assert "1" not in [m.id for m in matches]
    assert {m.metadata["category"] for m in matches} == {"billing"}


# Scripts


def run(args, env=None):
    env = {**os.environ, "VECTOR_STORE": "memory", "PINECONE_API_KEY": "", **(env or {})}
    completed = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=300)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_runs(script):
    assert run([str(script)]).strip()


def test_cli():
    output = run(["-m", "helpdesk_search", "I was charged twice", "--top-k", "1"])
    assert "[ 2-1]  Refunds and duplicate charges > Charged twice" in output
    output = run(["-m", "helpdesk_search", "money back", "--articles", "--category", "billing", "--top-k", "1"])
    assert "Refunds and duplicate charges" in output


# Pinecone, only when asked for: RUN_PINECONE_TESTS=1 pytest -m pinecone

pinecone_enabled = pytest.mark.skipif(
    not (os.getenv("RUN_PINECONE_TESTS") and os.getenv("PINECONE_API_KEY")),
    reason="set RUN_PINECONE_TESTS=1 and PINECONE_API_KEY to test against Pinecone",
)


@pytest.mark.pinecone
@pinecone_enabled
def test_pinecone_store_round_trip():
    store = PineconeStore("helpdesk-search-minilm", dimension(), namespace=f"test-{uuid.uuid4().hex[:8]}")
    try:
        records = load_sections()
        index_records(store, records)
        assert store.count() == len(records)
        assert search(store, "I was charged twice", top_k=1)[0].id == "2-1"
        assert search(store, "delete", top_k=3, filter={"category": "privacy"})[0].metadata["category"] == "privacy"

        store.update("2-1", metadata={"read_minutes": 9})
        store.wait_until(lambda: store.fetch(["2-1"])["2-1"][1]["read_minutes"] == 9)
        store.delete(["2-1"])
        # Stats and fetch catch up separately, so wait for both.
        store.wait_until(lambda: store.fetch(["2-1"]) == {} and store.count() == len(records) - 1)
    finally:
        store.delete_all()


@pytest.mark.pinecone
@pinecone_enabled
def test_pinecone_examples_run():
    for name in ["pinecone_basics.py", "update_and_delete.py", "metadata_filters.py"]:
        assert run([str(ROOT / "examples" / name)], env={"VECTOR_STORE": "pinecone", "PINECONE_API_KEY": os.environ["PINECONE_API_KEY"]})
