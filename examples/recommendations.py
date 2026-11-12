"""A "related articles" box: recommend items close to one the reader is viewing.

The query is the stored vector of the current article (query by id), so no
text is embedded at request time.
"""

from helpdesk_search.search import describe, open_index, related

store = open_index("articles")

for article_id in ["1", "7", "9"]:
    current = store.fetch([article_id])[article_id][1]
    print(f"Reading: {current['article_title']}")
    for match in related(store, article_id, top_k=2):
        print("  related " + describe(match))
    same_category = related(store, article_id, top_k=1, filter={"category": current["category"]})
    print("  same category only: " + (describe(same_category[0]) if same_category else "none"))
    print()
