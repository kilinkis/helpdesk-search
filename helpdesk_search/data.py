"""Load the help center CSV and turn it into records ready to embed.

The CSV has one row per section, with the article's fields repeated on each row.
It can be indexed at two levels:

- articles: one record per article, embedded from its title, summary and all
  of its sections.
- sections: one record per section, embedded from the article title plus the
  section, so a short section still carries the context of its article.

Every record gets an id, the text to embed, and metadata stored next to the
vector (returned with search results and usable in filters).
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "help_center.csv"


def load_help_center(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """The raw CSV, one row per section."""
    return pd.read_csv(path)


def load_articles(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """One record per article. The id is the article id, for example "2"."""
    rows = load_help_center(path)
    bodies = rows.groupby("article_id")[["section_title", "section_text"]].apply(
        lambda sections: " ".join(sections["section_title"] + ": " + sections["section_text"])
    )
    articles = rows.drop_duplicates("article_id").reset_index(drop=True)
    articles["unique_id"] = articles["article_id"].astype(str)
    # Title and summary alone are too short to say much. With the full body, articles
    # that cover related things end up close together, which recommendations rely on.
    # MiniLM reads at most 256 tokens; longer text is cut off, so keep articles short.
    articles["text"] = (
        articles["article_title"] + ". " + articles["article_summary"] + " " + articles["article_id"].map(bodies)
    )
    articles["metadata"] = articles.apply(
        lambda row: {
            "article_title": row["article_title"],
            "category": row["category"],
            "summary": row["article_summary"],
            "read_minutes": int(row["read_minutes"]),
        },
        axis=1,
    )
    return articles[["unique_id", "text", "metadata"]]


def load_sections(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """One record per section. The id joins article and section, for example "2-1"."""
    sections = load_help_center(path)
    sections["unique_id"] = sections["article_id"].astype(str) + "-" + sections["section_id"].astype(str)
    sections["text"] = sections.apply(section_text, axis=1)
    sections["metadata"] = sections.apply(
        lambda row: {
            "article_id": str(row["article_id"]),
            "article_title": row["article_title"],
            "section_title": row["section_title"],
            "category": row["category"],
            "text": row["section_text"],
            "read_minutes": int(row["read_minutes"]),
        },
        axis=1,
    )
    return sections[["unique_id", "text", "metadata"]]


def section_text(row: pd.Series) -> str:
    """The text embedded for a section: article title, section title and body.

    The article summary is left out on purpose. It is the same for every section
    of an article, so including it pulls all of them towards the same point and
    makes it harder to tell the sections apart.
    """
    return f"{row['article_title']}. {row['section_title']}. {row['section_text']}"
