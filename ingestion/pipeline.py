"""
ingestion/pipeline.py
──────────────────────
LangChain-based ingestion pipeline.

Flow:
  S3/R2 JSON  →  LangChain Documents  →  Text Splitter  →  Chunked Documents

Each JobListing becomes a LangChain Document with rich metadata,
then gets chunked so the embedding model handles it well.
"""

from __future__ import annotations
import json
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ─── Document Builder ──────────────────────────────────────────────────────────

def job_to_document(job: dict) -> Document:
    """
    Convert a raw job dict into a LangChain Document.

    The page_content is a structured text representation that's
    optimised for semantic search — concise, information-dense.
    """
    skills_str = ", ".join(job.get("skills", [])) or "Not specified"
    salary_str = job.get("salary") or "Not specified"

    page_content = f"""Job Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Location: {job.get('location', 'N/A')}
Salary: {salary_str}
Required Skills: {skills_str}

Job Description:
{job.get('description', '').strip()}
"""

    metadata: dict[str, Any] = {
        "job_id":     job.get("job_id", ""),
        "title":      job.get("title", ""),
        "company":    job.get("company", ""),
        "location":   job.get("location", ""),
        "salary":     salary_str,
        "skills":     skills_str,
        "job_url":    job.get("job_url", ""),
        "scraped_at": job.get("scraped_at", ""),
        "source":     job.get("source", "indeed"),
    }

    return Document(page_content=page_content, metadata=metadata)


# ─── Ingestion Pipeline ────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Loads job JSON → builds LangChain Documents → chunks them.

    Args:
        chunk_size:    Max chars per chunk (Mistral embed handles ~8k tokens,
                       but shorter chunks give better retrieval precision)
        chunk_overlap: Overlap between consecutive chunks
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_from_file(self, path: str) -> list[dict]:
        """Load job dicts from a local JSON file."""
        with open(path) as f:
            return json.load(f)

    def load_from_s3_data(self, data: list[dict]) -> list[dict]:
        """Accept already-downloaded S3 data (list of dicts)."""
        return data

    def build_documents(self, jobs: list[dict]) -> list[Document]:
        """Convert raw job dicts to LangChain Documents."""
        docs = [job_to_document(j) for j in jobs]
        logger.info(f"Built {len(docs)} Documents from {len(jobs)} job listings.")
        return docs

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        """
        Split Documents into smaller chunks.
        Metadata is preserved on every chunk — crucial for retrieval attribution.
        """
        chunks = self.splitter.split_documents(docs)
        logger.info(f"Chunked {len(docs)} docs → {len(chunks)} chunks.")
        return chunks

    def run(self, jobs: list[dict]) -> list[Document]:
        """Full ingestion: jobs → chunks (one-liner convenience)."""
        docs = self.build_documents(jobs)
        return self.chunk_documents(docs)
