"""Index whole articles: load the CSV, build the text and metadata, embed, upsert, search."""

from helpdesk_search.data import load_articles, load_help_center
from helpdesk_search.embeddings import dimension
from helpdesk_search.search import describe, index_records, search
from helpdesk_search.store import get_store

raw = load_help_center()
print(f"CSV: {len(raw)} rows (one per section), columns {list(raw.columns)}")

# One record per article: an id, the text to embed, and metadata stored with the vector.
articles = load_articles()
first = articles.iloc[0]
print(f"\n{len(articles)} article records. The first one:")
print(f"  unique_id: {first['unique_id']}")
print(f"  text:      {first['text']}")
print(f"  metadata:  {first['metadata']}")

store = get_store(dimension(), name="helpdesk-search-minilm", namespace="example-articles")
store.delete(articles["unique_id"].tolist())  # start clean when it's a Pinecone namespace
print(f"\nUpserted {index_records(store, articles)} vectors of {store.dimension} dimensions.")

for question in ["I was charged twice", "I can't log in", "Can I use the app without internet?"]:
    print(f"\n{question}")
    for match in search(store, question, top_k=2):
        print("  " + describe(match))
