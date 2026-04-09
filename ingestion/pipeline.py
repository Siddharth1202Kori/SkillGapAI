"""
ingestion/pipeline.py
──────────────────────
LangChain-based ingestion pipeline featuring Semantic Section Chunking & Metadata Enrichment.

Flow:
  S3/R2 JSON  →  Dynamic Regex Enriched Data  → Semantic Section Splitter →  LangChain Documents
"""

from __future__ import annotations
import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ─── Metadata Enrichment ───────────────────────────────────────────────────────

def _extract_rich_metadata(title: str, text: str) -> dict[str, str]:
    """Parse sophisticated metadata fields out of raw job descriptions via regex."""
    text_lower = (title + " " + text).lower()
    
    # 1. Seniority Level Extraction
    seniority = "Mid-Level"
    if any(w in text_lower for w in ["lead", "principal", "staff"]):
        seniority = "Lead/Principal"
    elif any(w in text_lower for w in ["director", "vp", "head of"]):
        seniority = "Director/VP"
    elif any(w in text_lower for w in ["senior", "sr.", " sr "]):
        seniority = "Senior"
    elif any(w in text_lower for w in ["junior", "jr.", "entry level", "entry-level"]):
        seniority = "Junior"

    # 2. Work Type Extraction
    work_type = "Remote" # Default context for these specific API boards
    if "hybrid" in text_lower:
        work_type = "Hybrid"
    elif "on-site" in text_lower or "onsite" in text_lower:
        work_type = "Onsite"
        
    # 3. Salary Band Extraction ($XX,XXX - $YY,YYY)
    salary_match = re.search(r'\$[\d,kK]+\s*(?:-|to)\s*\$[\d,kK]+', text, re.IGNORECASE)
    salary_band = salary_match.group(0) if salary_match else "Not specified"
    
    # 4. Required Years of Experience (e.g. "5+ years experience")
    exp_match = re.search(r'(\d+)\+?\s*years?(?:\s+of)?\s+(?:experience|exp)', text, re.IGNORECASE)
    years_exp = f"{exp_match.group(1)}+ years" if exp_match else "Not specified"
    
    return {
        "seniority_level": seniority,
        "work_type": work_type,
        "salary_parsed": salary_band,
        "required_years_exp": years_exp
    }


# ─── Semantic Document Builder ─────────────────────────────────────────────────

def job_to_documents(job: dict) -> list[Document]:
    """
    Transforms a raw job listing into *multiple* LangChain Documents using
    Semantic Section Chunking. It dynamically splits paragraphs whenever it 
    detects a new header (like "Requirements:" or "Benefits:").
    """
    description = job.get('description', '').strip()
    title = job.get("title", "")
    company = job.get("company", "")
    
    enriched = _extract_rich_metadata(title, description)
    
    # Fallback to API salary field if regex failed to pull one
    if job.get("salary") and enriched["salary_parsed"] == "Not specified":
        enriched["salary_parsed"] = job.get("salary")
        
    base_metadata: dict[str, Any] = {
        "job_id":             job.get("job_id", ""),
        "title":              title,
        "company":            company,
        "location":           job.get("location", ""),
        "skills":             ", ".join(job.get("skills", [])),
        "source":             job.get("source", "unknown"),
        "seniority_level":    enriched["seniority_level"],
        "work_type":          enriched["work_type"],
        "salary_band":        enriched["salary_parsed"],
        "required_years_exp": enriched["required_years_exp"]
    }
    
    # ── Semantic Chunking ──
    # Regex detecting common Markdown & Text standard headers
    header_pattern = re.compile(
        r'(?i)^[\*\#\-\s]*(requirements|responsibilities|qualifications|what you\'?ll do|what you will do|about the role|nice to have|benefits|perks)[\s\:\*]*\n', 
        re.MULTILINE
    )
    
    sections = []
    last_idx = 0
    current_header = "Overview"
    
    for match in header_pattern.finditer(description):
        start = match.start()
        if start > last_idx:
            chunk_text = description[last_idx:start].strip()
            if len(chunk_text) > 50:
                sections.append((current_header, chunk_text))
        last_idx = match.end()
        current_header = match.group(1).title()
        
    final_chunk = description[last_idx:].strip()
    if len(final_chunk) > 50:
        sections.append((current_header, final_chunk))
        
    if not sections:
        sections = [("Overview", description)]
        
    # Generate the finalized list of context-enriched Vector points
    docs = []
    for section_name, section_text in sections:
        # Prepend massive contextual anchoring so the LLM remembers which job 
        # this chunk belongs to when retrieved independently!
        page_content = f"[{title} at {company} - {section_name}]\n{section_text}"
        
        chunk_meta = base_metadata.copy()
        chunk_meta["section"] = section_name
        docs.append(Document(page_content=page_content, metadata=chunk_meta))
        
    return docs


# ─── Ingestion Pipeline ────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Loads job JSON → extracts rich metadata → segments via semantic layout headers → 
    and wraps with a fallback character text splitter for massive unbroken texts.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        # Fallback splitter is slightly larger since our Semantic logic does the heavy lifting now
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_from_file(self, path: str) -> list[dict]:
        """Load job dicts from a local JSON file."""
        with open(path) as f:
            return json.load(f)

    def load_from_s3_data(self, data: list[dict]) -> list[dict]:
        return data

    def build_documents(self, jobs: list[dict]) -> list[Document]:
        """Convert raw job dicts to context-enriched LangChain Documents."""
        docs = []
        for j in jobs:
            docs.extend(job_to_documents(j))
            
        logger.info(f"Built {len(docs)} Semantic Section Documents from {len(jobs)} total job listings.")
        return docs

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        """
        Applies a fallback size-splitter so we don't crash embeddings on massive sections.
        """
        chunks = self.fallback_splitter.split_documents(docs)
        logger.info(f"Final Fallback Chunking: {len(docs)} semantic sections → {len(chunks)} final vector chunks.")
        return chunks

    def run(self, jobs: list[dict]) -> list[Document]:
        docs = self.build_documents(jobs)
        return self.chunk_documents(docs)
