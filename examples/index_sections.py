"""Index sections instead of whole articles, and compare what each level finds.

An article vector is a blend of everything in the article. A section vector is
specific, so a question about one detail lands on the section that answers it.
"""

from helpdesk_search.data import load_articles, load_sections
from helpdesk_search.embeddings import dimension
from helpdesk_search.search import describe, index_records, search
from helpdesk_search.store import get_store

sections = load_sections()
first = sections.iloc[0]
print(f"{len(sections)} section records. The first one:")
print(f"  unique_id: {first['unique_id']}   (article id - section id)")
print(f"  text:      {first['text']}")
print(f"  metadata:  {sorted(first['metadata'])}")

article_store = get_store(dimension(), name="helpdesk-search-minilm", namespace="example-level-articles")
section_store = get_store(dimension(), name="helpdesk-search-minilm", namespace="example-level-sections")
for store, records in [(article_store, load_articles()), (section_store, sections)]:
    store.delete(records["unique_id"].tolist())
    index_records(store, records)

for question in ["How long does the reset link last?", "What happens to my projects if someone leaves?"]:
    print(f"\n{question}")
    print("  articles: " + describe(search(article_store, question, top_k=1)[0]))
    best = search(section_store, question, top_k=1)[0]
    print("  sections: " + describe(best))
    print(f"            {best.metadata['text']}")
