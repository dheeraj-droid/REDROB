#!/usr/bin/env python3
"""
WorthyHire — Model Downloader

This script pre-downloads the required HuggingFace models to the local cache.
Run this script BEFORE the main ranking pipeline to ensure that all models
are available when the network is disabled (as required by the submission spec).

Models downloaded:
1. BAAI/bge-small-en-v1.5 (~133MB) - Bi-encoder for semantic similarity
2. cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB) - Cross-encoder for reranking
"""

import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# Ignore symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def download_models():
    print("=" * 60)
    print("  WorthyHire — Pre-downloading models for offline use")
    print("=" * 60)
    
    print("\n[1/2] Downloading BAAI/bge-small-en-v1.5...")
    # This downloads and caches the model
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("  ✓ Bi-encoder model cached successfully.")
    
    print("\n[2/2] Downloading cross-encoder/ms-marco-MiniLM-L-6-v2...")
    # This downloads and caches the cross-encoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  ✓ Cross-encoder model cached successfully.")
    
    print("\n" + "=" * 60)
    print("  [OK] All models cached. You can now run the ranker safely")
    print("       with the network disabled.")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
