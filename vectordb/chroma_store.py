"""
vectordb/chroma_store.py
─────────────────────────
ChromaDB vector store: ingest chunks + similarity search.

Uses persistent local storage by default.
For production, swap PersistentClient for HttpClient pointing at
a hosted Chroma instance or Chroma Cloud.
"""

from __future__ import annotations
import os
from typing import Optional

import chromadb
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from dotenv import load_dotenv

from embeddings.mistral_embeddings import MistralEmbeddings

load_dotenv()


class ChromaVectorStore:
    """
    Manages a persistent ChromaDB collection of job embeddings.

    Args:
        persist_dir:       Directory to persist ChromaDB data
        collection_name:   Chroma collection name
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "indeed_jobs"
        )
        self.embeddings = MistralEmbeddings()
        self._store: Optional[Chroma] = None
        logger.info(
            f"ChromaVectorStore: collection='{self.collection_name}', "
            f"persist_dir='{self.persist_dir}'"
        )

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._store

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[Document], batch_size: int = 100):
        """
        Add chunked Documents to ChromaDB.
        Batches to avoid memory pressure on large datasets.
        """
        store = self._get_store()
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            store.add_documents(batch)
            logger.info(f"  Indexed batch {i // batch_size + 1}: {len(batch)} chunks ({i + len(batch)}/{total})")

        logger.success(f"✓ Indexed {total} chunks into '{self.collection_name}'")

    def reset_collection(self):
        """Wipe and recreate the collection (useful for re-ingestion)."""
        client = PersistentClient(path=self.persist_dir)
        try:
            client.delete_collection(self.collection_name)
            logger.warning(f"Deleted collection '{self.collection_name}'")
        except Exception:
            pass
        self._store = None  # Force reinit

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Semantic similarity search.

        Args:
            query:  Natural language query
            k:      Number of results to return
            filter: Optional metadata filter, e.g. {"location": "remote"}

        Returns:
            List of matching Document chunks with metadata
        """
        store = self._get_store()
        results = store.similarity_search(query, k=k, filter=filter)
        logger.debug(f"similarity_search('{query[:50]}...') → {len(results)} results")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Returns (Document, cosine_distance) pairs — lower = more similar."""
        store = self._get_store()
        return store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 6):
        """Return a LangChain-compatible retriever for use in chains."""
        return self._get_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    @property
    def count(self) -> int:
        """Number of chunks currently indexed."""
        client = PersistentClient(path=self.persist_dir)
        collection = client.get_collection(self.collection_name)
        return collection.count()
