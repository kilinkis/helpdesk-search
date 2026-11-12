"""Semantic search over a help center with sentence-transformers and Pinecone."""

from dotenv import load_dotenv

# VECTOR_STORE, PINECONE_API_KEY and PINECONE_INDEX can live in a .env file.
# Variables already set in the environment win.
load_dotenv()
