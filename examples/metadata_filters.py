"""Combine meaning with exact conditions: metadata filters narrow the search before ranking."""

from helpdesk_search.search import describe, open_index, search

store = open_index("sections")
question = "how do I delete something"

filters = {
    "no filter": None,
    "category is account": {"category": "account"},
    "category is billing or privacy": {"category": {"$in": ["billing", "privacy"]}},
    "short reads (3 minutes or less)": {"read_minutes": {"$lte": 3}},
    "technical and longer than 4 minutes": {"$and": [{"category": "technical"}, {"read_minutes": {"$gt": 4}}]},
}

print(question)
for label, filter in filters.items():
    print(f"\n  {label}: {filter}")
    for match in search(store, question, top_k=2, filter=filter):
        print(f"    {describe(match)}  ({match.metadata['category']}, {match.metadata['read_minutes']} min)")
