"""
download_models.py
──────────────────
Pre-downloads the CrossEncoder model ONCE at build time (not at app startup).
Called by Render's Build Command before the gunicorn server starts.
This avoids a 90MB download on every cold start.
"""
from sentence_transformers import CrossEncoder

print("Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
print("Model cached successfully. Build step complete.")
