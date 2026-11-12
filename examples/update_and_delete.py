"""Keep the index in sync with the help center: update metadata, re-embed changed text, delete."""

from helpdesk_search.data import load_articles
from helpdesk_search.embeddings import dimension, embed
from helpdesk_search.search import describe, index_records, search
from helpdesk_search.store import get_store

articles = load_articles()
store = get_store(dimension(), name="helpdesk-search-minilm", namespace="example-updates")
store.delete(articles["unique_id"].tolist())
index_records(store, articles)


def show(question):
    print(f"  {question}")
    for match in search(store, question, top_k=2):
        print(f"    {describe(match)}  ({match.metadata['read_minutes']} min)")


print("Before:")
show("Where can I see the API key?")

# 1. Metadata only: no new embedding needed.
store.update("9", metadata={"read_minutes": 7})

# 2. The text changed, so the vector must be recomputed. Same id = the old vector is replaced.
new_title = "Developer API and data export"
new_text = f"{new_title}. Create and revoke API keys, call the REST API from scripts, and export your workspace."
store.update("9", vector=embed([new_text])[0], metadata={"article_title": new_title})
store.wait_until(lambda: store.fetch(["9"]).get("9", (None, {}))[1].get("article_title") == new_title)

print("\nAfter updating article 9's metadata and re-embedding its new text:")
show("Where can I see the API key?")

# 3. Delete an article that was retired.
store.delete(["10"])
store.wait_until(lambda: not store.fetch(["10"]) and store.count() == len(articles) - 1)
print(f"\nDeleted article 10. The index now has {store.count()} vectors.")
print("fetch(['10']) ->", store.fetch(["10"]))
