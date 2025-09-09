# RecTask – Content-based Recommender with Sentence Embeddings

This repository provides a simple content-based recommendation system using sentence embeddings from sentence-transformers/paraphrase-MPNet-base-v2.

Features:
- Weighted embeddings from separate encodes of Title (0.3) and Genres (0.7), then per-item L2 normalization
- Cosine similarity recommendations
- Single-file cache (embeddings_cache.npz) with per-item hashes to skip re-embedding unchanged items
- Outputs include Id, Title, Genres, Score

Setup:
1) Python env and dependencies:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install sentence-transformers numpy scikit-learn

2) Data:
   Provide a JSON file similar to all 1.json (list of items or {"items": [...]}) with keys:
   - Id (string/int)
   - Title (string)
   - Genres (list of dicts with Name, or list of strings, or string)

Run:
source .venv/bin/activate
python recommend.py --data "/path/to/all+1.json" --k 10
Optional seed:
python recommend.py --data "/path/to/all+1.json" --seed 317111 --k 10

Notes:
- If --seed is omitted, the first available item is used.
- A cache file embeddings_cache.npz is created next to the script. Delete it to force full re-embed.
- The cache automatically invalidates when Title/Genres, model name, or weights change.
