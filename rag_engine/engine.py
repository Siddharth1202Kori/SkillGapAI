"""
rag_engine/engine.py
──────────────────────
Core RAG engine: retrieves relevant job chunks from ChromaDB,
then uses Mistral LLM to generate:

  1. Skill Gap Analysis  — what skills the market demands vs user's profile
  2. Learning Suggestions — concrete resources to close each gap
  3. Job Match Summary   — top matching roles with fit explanation
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from mistralai import Mistral
from langchain_core.documents import Document
from loguru import logger
from dotenv import load_dotenv

from vectordb.chroma_store import ChromaVectorStore

load_dotenv()


# ─── Output Models ─────────────────────────────────────────────────────────────

@dataclass
class SkillGapResult:
    query: str
    matched_jobs: list[dict]          # [{title, company, location, url, score}]
    in_demand_skills: list[str]       # skills seen across retrieved jobs
    skill_gaps: list[str]             # skills user lacks
    suggestions: list[dict]           # [{skill, resource, type}]
    raw_analysis: str                 # full LLM narrative


# ─── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine for job skill analysis.

    Args:
        vector_store: ChromaVectorStore instance
        model:        Mistral model for generation (default: mistral-small-latest)
        k:            Number of chunks to retrieve per query
    """

    GENERATION_MODEL = "mistral-small-latest"  # swap for mistral-large-latest for better quality

    SYSTEM_PROMPT = """You are a career advisor and technical skills analyst.
You are given job listings retrieved from Indeed, and a user's background/query.
Your job is to:
1. Identify the most in-demand skills across the retrieved job listings
2. Identify skill gaps based on the user's stated background
3. Suggest specific, actionable learning resources to close each gap
4. Highlight the top matching job roles

Be specific, concise, and actionable. Format your response clearly with sections."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        model: str = GENERATION_MODEL,
        k: int = 8,
    ):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set.")
        self.client = Mistral(api_key=api_key)
        self.store = vector_store
        self.model = model
        self.k = k
        logger.info(f"RAGEngine ready (model={model}, k={k})")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve top-k relevant job chunks with similarity scores."""
        return self.store.similarity_search_with_score(query, k=self.k)

    def _format_context(self, results: list[tuple[Document, float]]) -> str:
        """Format retrieved chunks into an LLM-readable context block."""
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            parts.append(
                f"--- Job {i} (relevance score: {1 - score:.2f}) ---\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Company: {meta.get('company', 'N/A')}\n"
                f"Location: {meta.get('location', 'N/A')}\n"
                f"Skills: {meta.get('skills', 'N/A')}\n"
                f"Content:\n{doc.page_content}\n"
            )
        return "\n".join(parts)

    def _extract_matched_jobs(self, results: list[tuple[Document, float]]) -> list[dict]:
        """Deduplicate and summarise matched jobs from retrieved chunks."""
        seen = set()
        jobs = []
        for doc, score in results:
            meta = doc.metadata
            job_id = meta.get("job_id", "")
            if job_id not in seen:
                seen.add(job_id)
                jobs.append({
                    "title":    meta.get("title", "N/A"),
                    "company":  meta.get("company", "N/A"),
                    "location": meta.get("location", "N/A"),
                    "url":      meta.get("job_url", ""),
                    "skills":   meta.get("skills", ""),
                    "score":    round(1 - score, 3),
                })
        return jobs

    # ── Generation ─────────────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        query: str,
        user_background: Optional[str],
        context: str,
    ) -> str:
        background_section = (
            f"\n\nUser Background:\n{user_background}" if user_background else ""
        )
        return f"""User Query: {query}{background_section}

Retrieved Job Listings:
{context}

Please provide:
## 1. In-Demand Skills
List the top skills appearing across these job listings.

## 2. Skill Gap Analysis
Based on the user's background, identify which required skills they likely lack.
If no background is provided, list the most specialised/advanced skills that are commonly missing.

## 3. Learning Suggestions
For each skill gap, suggest:
- A specific course, tutorial, or resource (with platform name)
- Estimated time to basic proficiency

## 4. Top Matching Roles
List the 3 most relevant job titles from the listings and explain why they match.

## 5. Action Plan
A 3-step prioritised action plan for the user.
"""

    def analyze(
        self,
        query: str,
        user_background: Optional[str] = None,
    ) -> SkillGapResult:
        """
        Full RAG pipeline: query → retrieve → generate skill gap analysis.

        Args:
            query:           e.g. "machine learning engineer jobs"
            user_background: User's skills/experience as free text (optional)

        Returns:
            SkillGapResult with structured + narrative output
        """
        logger.info(f"RAG analyze: '{query}'")

        # 1. Retrieve
        results = self.retrieve(query)
        if not results:
            logger.warning("No results retrieved — collection may be empty.")
            return SkillGapResult(
                query=query, matched_jobs=[], in_demand_skills=[],
                skill_gaps=[], suggestions=[], raw_analysis="No job data found in vector store."
            )

        # 2. Build context
        context = self._format_context(results)
        matched_jobs = self._extract_matched_jobs(results)

        # 3. Generate
        user_prompt = self._build_user_prompt(query, user_background, context)

        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        raw_analysis = response.choices[0].message.content
        logger.success("LLM analysis complete.")

        # 4. Extract structured data from matched jobs
        all_skills = set()
        for job in matched_jobs:
            for skill in job["skills"].split(", "):
                if skill.strip():
                    all_skills.add(skill.strip().lower())

        return SkillGapResult(
            query=query,
            matched_jobs=matched_jobs,
            in_demand_skills=sorted(all_skills),
            skill_gaps=[],       # populated by LLM narrative in raw_analysis
            suggestions=[],      # populated by LLM narrative in raw_analysis
            raw_analysis=raw_analysis,
        )
