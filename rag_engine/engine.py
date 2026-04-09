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

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

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
    Retrieval-Augmented Generation engine featuring Hybrid Search (Dense+BM25) 
    and Cross-Encoder reranking.
    """

    GENERATION_MODEL = "mistral-small-latest"

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
        
        # Load lightning-fast generic reranker pipeline from Sentence Transformers
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info(f"RAGEngine ready (model={model}, k={k}, Hybrid Search + CrossEncoder Active)")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve using Hybrid Search (BM25 + Dense) -> RRF Fusion -> CrossEncoder Rerank."""
        fetch_k = self.k * 3
        logger.info(f"Hybrid Search: Fetching top {fetch_k} dense and sparse candidates...")
        
        # 1. DENSE SEARCH (ChromaDB Vector Embeddings)
        dense_tuples = self.store.similarity_search_with_score(query, k=fetch_k)
        
        # 2. SPARSE SEARCH (BM25 Keyword Matching)
        all_docs = self.store.get_all_documents()
        if not all_docs:
            return []
            
        tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Slice the strongest BM25 integer indices
        sparse_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        sparse_docs = [all_docs[i] for i in sparse_indices]
        
        # 3. RECIPROCAL RANK FUSION (RRF)
        # Prevents either search algorithm from dominating by merging absolute ranks
        rrf_k = 60
        scores = {}
        doc_map = {}
        
        def get_doc_id(d):
            return getattr(d, 'id', hash(d.page_content)) if getattr(d, 'id', None) else hash(d.page_content)
            
        for rank, (doc, _score) in enumerate(dense_tuples):
            d_id = get_doc_id(doc)
            doc_map[d_id] = doc
            scores[d_id] = scores.get(d_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        for rank, doc in enumerate(sparse_docs):
            d_id = get_doc_id(doc)
            doc_map[d_id] = doc
            scores[d_id] = scores.get(d_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # Siphon the fused highest-confidence candidate pool (max 20 candidates per best practice)
        fused_pool = [doc_map[d_id] for d_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]]
        
        # 4. CROSS-ENCODER RERANKING
        logger.info(f"Reranking top {len(fused_pool)} candidates using MS-MARCO CrossEncoder...")
        cross_inp = [[query, doc.page_content] for doc in fused_pool]
        rerank_scores = self.reranker.predict(cross_inp)
        
        # Combine documents with rerank scoring mathematically
        results = list(zip(fused_pool, rerank_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract ultimate 'Top K' payload for Generation framework
        final_top = results[:self.k]
        logger.success(f"Retrieved and natively reranked top {len(final_top)} context chunks.")
        
        return [(doc, float(s)) for doc, s in final_top]

    def _format_context(self, results: list[tuple[Document, float]]) -> str:
        """Format retrieved chunks into an LLM-readable context block."""
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            parts.append(
                f"--- Job {i} (relevance score: {score:.2f}) ---\n"
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
                    "score":    round(score, 3), # Logit confidence scores
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
