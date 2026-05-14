#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indexer import inverted_index, pagerank

if __name__ == "__main__":
    print("=== Building inverted index + BM25 stats ===")
    inverted_index.build()
    print("\n=== Computing PageRank ===")
    pagerank.compute()
    print("\nDone.")
