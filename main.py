"""
main.py
────────
Orchestrates the full RAG pipeline:

  Step 1 → Scrape Indeed  (or load from file / Supabase)
  Step 2 → Upload to Supabase Storage
  Step 3 → Ingest + Chunk via LangChain
  Step 4 → Embed with Mistral + Index in ChromaDB
  Step 5 → Query the RAG engine for skill gap analysis

Usage:
  # Full pipeline (scrape + ingest + query)
  python main.py --query "machine learning engineer" --location "remote" --user-background "I know Python and pandas"

  # Skip scraping, use existing local JSON
  python main.py --load-file jobs.json --query "data engineer"

  # Skip scraping, use existing Supabase file
  python main.py --load-supabase jobs/2024-01-15_ml.json --query "mlops engineer"

  # Query only (ChromaDB already populated)
  python main.py --query-only --query "backend engineer fastapi"
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Internal imports ────────────────────────────────────────────────────────────
from scraper.remotive_scraper import RemotiveScraper
from utils.cloud_storage import CloudStorage
from ingestion.pipeline import IngestionPipeline
from vectordb.chroma_store import ChromaVectorStore
from rag_engine.engine import RAGEngine


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Job RAG Pipeline (Remotive + Indeed)")
    p.add_argument("--query",    required=True, help="Search query for jobs + RAG question")
    p.add_argument("--location", default="remote", help="Job location filter")
    p.add_argument("--pages",    type=int, default=3, help="Max pages to scrape (Indeed only)")
    p.add_argument("--user-background", default=None,
                   help="Your skills/experience (free text) for gap analysis")

    # Data source options (mutually exclusive)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--load-file",  help="Skip scraping; load jobs from local JSON file")
    source.add_argument("--load-supabase", help="Skip scraping; load jobs from Supabase")
    source.add_argument("--query-only", action="store_true",
                        help="Skip scraping + ingestion; query existing ChromaDB")

    # Scraping Configuration
    p.add_argument("--remotive-category", default=None,
                   help="Remotive category slug (e.g. 'software-dev', 'ai-ml', 'data')")

    # Storage options
    p.add_argument("--no-supabase", action="store_true", help="Skip Supabase upload")
    p.add_argument("--reset-db", action="store_true", help="Wipe ChromaDB before ingesting")

    return p.parse_args()


# ─── Pipeline Steps ────────────────────────────────────────────────────────────

def step_scrape(args) -> list[dict]:
    """Step 1: Scrape jobs (Remotive API)."""
    logger.info("🔍 Using Remotive API (no blocking)...")
    scraper = RemotiveScraper()
    listings = scraper.scrape(
        query=args.query,
        location=args.location,
        category=args.remotive_category,
    )

    return [asdict(l) for l in listings]


def step_load(args) -> list[dict]:
    """Step 1 (alt): Load from file or Supabase."""
    if args.load_file:
        logger.info(f"📂 Loading from file: {args.load_file}")
        with open(args.load_file) as f:
            return json.load(f)

    if args.load_supabase:
        logger.info(f"☁️  Loading from Supabase: {args.load_supabase}")
        storage = CloudStorage()
        return storage.download_jobs(args.load_supabase)

    return []


def step_upload_supabase(jobs: list[dict], query: str) -> str | None:
    """Step 2: Upload raw jobs to Supabase Storage."""
    try:
        storage = CloudStorage()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        safe_query = query.replace(" ", "_")[:40]
        key = f"jobs/{date_str}_{safe_query}.json"
        return storage.upload_jobs(jobs, key)
    except Exception as e:
        logger.warning(f"Supabase upload failed (continuing): {e}")
        return None


def step_ingest(jobs: list[dict], vector_store: ChromaVectorStore, reset: bool):
    """Steps 3 + 4: Ingest, chunk, embed, index."""
    if reset:
        logger.warning("🗑  Resetting ChromaDB collection...")
        vector_store.reset_collection()

    pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)
    chunks = pipeline.run(jobs)
    vector_store.add_documents(chunks)
    logger.success(f"✓ {len(chunks)} chunks indexed. Total in DB: {vector_store.count}")


def step_query(args, vector_store: ChromaVectorStore):
    """Steps 5 + 6: RAG query + output."""
    engine = RAGEngine(vector_store=vector_store)
    result = engine.analyze(
        query=args.query,
        user_background=args.user_background,
    )

    # ── Print Output ────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  🎯  SKILL GAP ANALYSIS & JOB RECOMMENDATIONS")
    print("═" * 70)

    print(f"\nQuery: {result.query}")

    if result.matched_jobs:
        print(f"\n📋 Top Matched Jobs ({len(result.matched_jobs)} retrieved):")
        for i, job in enumerate(result.matched_jobs[:5], 1):
            print(f"  {i}. {job['title']} @ {job['company']} — {job['location']}")
            print(f"     Score: {job['score']} | Skills: {job['skills'][:80]}...")
            if job["url"]:
                print(f"     URL: {job['url']}")

    if result.in_demand_skills:
        print(f"\n💡 In-Demand Skills Detected:")
        print("  " + ", ".join(result.in_demand_skills))

    print("\n" + "─" * 70)
    print(result.raw_analysis)
    print("─" * 70 + "\n")

    # Save output
    out_path = Path("output_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "query":           result.query,
            "matched_jobs":    result.matched_jobs,
            "in_demand_skills": result.in_demand_skills,
            "analysis":        result.raw_analysis,
        }, f, indent=2)
    logger.success(f"Analysis saved → {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    vector_store = ChromaVectorStore()

    # ── Data Acquisition ────────────────────────────────────────────────────────
    if args.query_only:
        logger.info("⏭  Skipping scrape + ingest (--query-only mode)")
    else:
        # Get jobs
        if args.load_file or args.load_supabase:
            jobs = step_load(args)
        else:
            jobs = step_scrape(args)

        if not jobs:
            logger.error("No jobs found — check your query or try --rss flag.")
            sys.exit(1)

        logger.info(f"✅ Got {len(jobs)} job listings.")

        # Save locally as backup
        # REMOVED: Saving jobs locally. Relying strictly on Supabase Cloud Storage
        # to prevent disk-bloat when deployed to hosting providers like Render!
        # Upload to Supabase (unless disabled)
        if not args.no_supabase:
            step_upload_supabase(jobs, args.query)

        # Ingest into ChromaDB
        step_ingest(jobs, vector_store, reset=args.reset_db)

    # ── Query ───────────────────────────────────────────────────────────────────
    step_query(args, vector_store)


if __name__ == "__main__":
    main()
