"""Search the help center from the terminal.

    python -m helpdesk_search "I was charged twice"
    python -m helpdesk_search "can't log in" --articles --top-k 2
    python -m helpdesk_search "how do refunds work" --category billing --model bert

Uses the in-memory store by default. Set VECTOR_STORE=pinecone to search a
Pinecone index instead (created and filled on first use).
"""

import argparse

from dotenv import load_dotenv

from helpdesk_search.embeddings import MODELS
from helpdesk_search.search import describe, open_index, search


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m helpdesk_search", description="Semantic search over the help center.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--category", help="only search this category, e.g. billing")
    parser.add_argument("--articles", action="store_true", help="search whole articles instead of sections")
    parser.add_argument("--model", choices=sorted(MODELS), default="minilm")
    args = parser.parse_args()

    store = open_index("articles" if args.articles else "sections", args.model)
    filter = {"category": args.category} if args.category else None
    for match in search(store, args.question, args.model, args.top_k, filter):
        print(describe(match))
        if "text" in match.metadata:
            print(f"                {match.metadata['text']}")


if __name__ == "__main__":
    main()
