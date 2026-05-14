#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embedder.embedder import build

if __name__ == "__main__":
    build()
