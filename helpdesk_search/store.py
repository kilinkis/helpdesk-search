"""Vector stores: the same small interface backed by memory or by Pinecone.

A store keeps records of (id, vector, metadata). Writing the same id again
replaces the record, which is what "upsert" means.

InMemoryStore follows Pinecone's behaviour closely enough (scores, metadata
filters, query by id) that every example runs offline with the same code.
"""

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

Record = tuple[str, Sequence[float], dict]
METRICS = ("cosine", "dotproduct")


@dataclass
class Match:
    id: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    dimension: int

    def upsert(self, records: Iterable[Record]) -> None: ...
    def query(
        self, vector: Sequence[float] | None = None, *, id: str | None = None, top_k: int = 3, filter: dict | None = None
    ) -> list[Match]: ...
    def fetch(self, ids: Sequence[str]) -> dict[str, tuple[list[float], dict]]: ...
    def update(self, id: str, *, vector: Sequence[float] | None = None, metadata: dict | None = None) -> None: ...
    def delete(self, ids: Sequence[str]) -> None: ...
    def count(self) -> int: ...
    def wait_until(self, check, timeout: float = 60) -> None: ...


class InMemoryStore:
    """A brute-force vector store: compares the query with every vector."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.dimension = dimension
        self.metric = metric
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    def upsert(self, records: Iterable[Record]) -> None:
        for id, vector, metadata in records:
            self._vectors[id] = self._check(vector)
            self._metadata[id] = dict(metadata)

    def query(self, vector=None, *, id=None, top_k=3, filter=None) -> list[Match]:
        if (vector is None) == (id is None):
            raise ValueError("pass either a vector or the id of a stored vector")
        query = self._vectors[id] if id is not None else self._check(vector)
        ids = [i for i in self._vectors if matches_filter(self._metadata[i], filter)]
        if not ids:
            return []
        matrix = np.stack([self._vectors[i] for i in ids])
        scores = matrix @ query
        if self.metric == "cosine":
            scores = scores / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        best = np.argsort(-scores)[:top_k]
        return [Match(ids[i], float(scores[i]), dict(self._metadata[ids[i]])) for i in best]

    def fetch(self, ids):
        return {i: (self._vectors[i].tolist(), dict(self._metadata[i])) for i in ids if i in self._vectors}

    def update(self, id, *, vector=None, metadata=None) -> None:
        if id not in self._vectors:
            return  # Pinecone also ignores updates to ids that don't exist
        if vector is not None:
            self._vectors[id] = self._check(vector)
        if metadata:
            self._metadata[id].update(metadata)  # merged, like Pinecone's set_metadata

    def delete(self, ids) -> None:
        for i in ids:
            self._vectors.pop(i, None)
            self._metadata.pop(i, None)

    def count(self) -> int:
        return len(self._vectors)

    def wait_until(self, check, timeout: float = 60) -> None:
        """Writes are visible immediately here, so there is nothing to wait for."""

    def _check(self, vector) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self.dimension,):
            raise ValueError(f"expected a vector of {self.dimension} numbers, got shape {array.shape}")
        return array


class PineconeStore:
    """A namespace in a Pinecone serverless index. Creates the index if it doesn't exist."""

    def __init__(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        from pinecone import Pinecone, ServerlessSpec

        self.dimension = dimension
        self.namespace = namespace
        self.client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        if not self.client.has_index(index_name):
            # Waits until the index is ready. Dimension and metric can't be changed later.
            self.client.create_index(
                name=index_name, dimension=dimension, metric=metric, spec=ServerlessSpec(cloud=cloud, region=region)
            )
        self.index = self.client.Index(index_name)

    def upsert(self, records: Iterable[Record], batch_size: int = 100) -> None:
        vectors = [{"id": i, "values": [float(x) for x in v], "metadata": m} for i, v, m in records]
        for start in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[start : start + batch_size], namespace=self.namespace, show_progress=False)

    def query(self, vector=None, *, id=None, top_k=3, filter=None) -> list[Match]:
        response = self.index.query(
            vector=[float(x) for x in vector] if vector is not None else None,
            id=id,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            namespace=self.namespace,
        )
        return [Match(m.id, m.score, dict(m.metadata or {})) for m in response.matches]

    def fetch(self, ids):
        response = self.index.fetch(ids=list(ids), namespace=self.namespace)
        return {i: (list(v.values), dict(v.metadata or {})) for i, v in response.vectors.items()}

    def update(self, id, *, vector=None, metadata=None) -> None:
        self.index.update(
            id=id,
            values=[float(x) for x in vector] if vector is not None else None,
            set_metadata=metadata,
            namespace=self.namespace,
        )

    def delete(self, ids) -> None:
        from pinecone.errors.exceptions import NotFoundError

        try:
            self.index.delete(ids=list(ids), namespace=self.namespace)
        except NotFoundError:
            pass  # the namespace doesn't exist yet, so there is nothing to delete

    def delete_all(self) -> None:
        if self.count():
            self.index.delete(delete_all=True, namespace=self.namespace)

    def count(self) -> int:
        namespaces = self.index.describe_index_stats().namespaces or {}
        summary = namespaces.get(self.namespace or "__default__")
        return summary.vector_count if summary else 0

    def wait_until(self, check, timeout: float = 60) -> None:
        """Pinecone is eventually consistent: a write can take a few seconds to show up in reads."""
        deadline = time.monotonic() + timeout
        while not check():
            if time.monotonic() > deadline:
                raise TimeoutError("Pinecone did not reflect the write in time")
            time.sleep(1)


def get_store(dimension: int, *, name: str = "helpdesk-search", namespace: str = "", backend: str | None = None):
    """The store selected by the VECTOR_STORE env var: "memory" (default) or "pinecone"."""
    backend = backend or os.getenv("VECTOR_STORE", "memory")
    if backend == "memory":
        return InMemoryStore(dimension)
    if backend == "pinecone":
        return PineconeStore(index_name=name, dimension=dimension, namespace=namespace)
    raise ValueError(f"Unknown VECTOR_STORE {backend!r}. Use 'memory' or 'pinecone'.")


# Metadata filters use Pinecone's syntax, for example
#   {"category": "billing"}
#   {"read_minutes": {"$lte": 3}}
#   {"$or": [{"category": "billing"}, {"category": "account"}]}
_COMPARISONS = {
    "$eq": lambda value, target: value == target,
    "$ne": lambda value, target: value != target,
    "$gt": lambda value, target: value is not None and value > target,
    "$gte": lambda value, target: value is not None and value >= target,
    "$lt": lambda value, target: value is not None and value < target,
    "$lte": lambda value, target: value is not None and value <= target,
    "$in": lambda value, target: value in target,
    "$nin": lambda value, target: value not in target,
}


def matches_filter(metadata: Mapping[str, Any], filter: Mapping[str, Any] | None) -> bool:
    if not filter:
        return True
    for key, condition in filter.items():
        if key == "$and":
            if not all(matches_filter(metadata, part) for part in condition):
                return False
        elif key == "$or":
            if not any(matches_filter(metadata, part) for part in condition):
                return False
        else:
            if not isinstance(condition, Mapping):
                condition = {"$eq": condition}
            for operator, target in condition.items():
                if operator == "$exists":
                    if (key in metadata) != target:
                        return False
                elif not _compare(operator, metadata.get(key), target):
                    return False
    return True


def _compare(operator: str, value: Any, target: Any) -> bool:
    if operator not in _COMPARISONS:
        raise ValueError(f"Unsupported filter operator {operator}")
    if isinstance(value, list):  # a list field matches if any of its items does
        if operator in ("$ne", "$nin"):
            return all(_COMPARISONS[operator](item, target) for item in value)
        return any(_COMPARISONS[operator](item, target) for item in value)
    return _COMPARISONS[operator](value, target)
