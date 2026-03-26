"""
embeddings/mistral_embeddings.py
──────────────────────────────────
Wraps Mistral's embedding API for use with LangChain + ChromaDB.

Model used: mistral-embed
  - 1024-dim dense vectors
  - Context window: 8192 tokens
  - Great for semantic search over technical/professional text
"""

from __future__ import annotations
import os
import time
from typing import List

from mistralai import Mistral
from langchain_core.embeddings import Embeddings
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class MistralEmbeddings(Embeddings):
    """
    LangChain-compatible Mistral embedding wrapper.
    Implements .embed_documents() and .embed_query() as required by LangChain.
    """

    MODEL = "mistral-embed"
    BATCH_SIZE = 32          # Mistral allows up to 512, but 32 is safe
    RATE_LIMIT_DELAY = 0.5   # seconds between batches

    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in environment.")
        self.client = Mistral(api_key=api_key)
        logger.info(f"MistralEmbeddings ready (model={self.MODEL})")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        response = self.client.embeddings.create(
            model=self.MODEL,
            inputs=texts,
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        Batches automatically to respect API limits.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            logger.debug(f"Embedding batch {i // self.BATCH_SIZE + 1} ({len(batch)} texts)...")
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

            if i + self.BATCH_SIZE < len(texts):
                time.sleep(self.RATE_LIMIT_DELAY)

        logger.info(f"Embedded {len(all_embeddings)} documents.")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        result = self._embed_batch([text])
        return result[0]
