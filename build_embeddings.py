"""
build_embeddings.py
-------------------
Build a local search index for ayushman_bharat.txt (TF-IDF, no Mistral API).

Usage:
    python build_embeddings.py

This is optional. bot.py builds the same index at startup if the cache is
missing. Re-run after you edit ayushman_bharat.txt, or just restart the bot.
"""
from pathlib import Path

from EXT.bot import EMBED_CACHE_FILE, EmbeddingRetriever, KNOWLEDGE_FILE


def main():
    print(f"Reading knowledge file: {KNOWLEDGE_FILE}")
    retriever = EmbeddingRetriever(KNOWLEDGE_FILE)
    print(f"  -> {len(retriever.chunks)} chunks indexed locally (no API calls)")
    print(f"\nSaving cache -> {EMBED_CACHE_FILE}")
    if Path(EMBED_CACHE_FILE).exists():
        print("\nDone. bot.py will load this cache at startup.")
        print("You can Ctrl+C the old Mistral embed script if it is still waiting.")
    else:
        raise SystemExit("ERROR: cache was not written")


if __name__ == "__main__":
    main()
