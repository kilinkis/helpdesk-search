"""The raw Pinecone client: create an index, upsert, query, fetch, inspect, delete.

Needs PINECONE_API_KEY (in the environment or .env). Creates a temporary
serverless index called "helpdesk-basics" and deletes it at the end.
The rest of the repo wraps these calls in helpdesk_search/store.py.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()
if not os.getenv("PINECONE_API_KEY"):
    print("Set PINECONE_API_KEY to run this example. Everything else runs without it.")
    raise SystemExit(0)

from pinecone import Pinecone, ServerlessSpec  # noqa: E402

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
name = "helpdesk-basics"

# An index stores vectors of one fixed dimension and compares them with one metric.
if not pc.has_index(name):
    pc.create_index(name=name, dimension=3, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
description = pc.describe_index(name)
print(f"Index '{name}': dimension {description.dimension}, metric {description.metric}, host {description.host}")
print("All indexes:", [index.name for index in pc.list_indexes()])

index = pc.Index(name)
try:
    # Each vector has an id, its values and optional metadata. Same id again = overwrite.
    index.upsert(
        vectors=[
            {"id": "a", "values": [1.0, 0.0, 0.0], "metadata": {"category": "billing"}},
            {"id": "b", "values": [0.9, 0.1, 0.0], "metadata": {"category": "billing"}},
            {"id": "c", "values": [0.0, 1.0, 0.0], "metadata": {"category": "account"}},
        ],
        namespace="demo",
        show_progress=False,
    )

    # Writes are eventually consistent, so wait until the stats show all three.
    while (stats := index.describe_index_stats()).total_vector_count < 3:
        time.sleep(1)
    print(f"\nStats: {stats.total_vector_count} vectors, namespaces {list(stats.namespaces)}")

    result = index.query(vector=[1.0, 0.05, 0.0], top_k=2, include_metadata=True, namespace="demo")
    print("\nquery([1, 0.05, 0]):")
    for match in result.matches:
        print(f"  {match.id}  score {match.score:.3f}  {match.metadata}")

    result = index.query(vector=[1.0, 0.05, 0.0], top_k=2, filter={"category": "account"}, namespace="demo")
    print("\nSame query, filter category == account:", [m.id for m in result.matches])

    fetched = index.fetch(ids=["c"], namespace="demo")
    print("\nfetch(['c']):", fetched.vectors["c"].values)
finally:
    pc.delete_index(name)
    print(f"\nDeleted index '{name}'.")
