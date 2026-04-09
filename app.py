import sys
import json
import os
import uuid
import threading
from datetime import datetime, timezone
from dataclasses import asdict

from flask import Flask, request, jsonify, send_from_directory
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='frontend')

# ── In-memory job store ──────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()

# ── Pipeline components (populated by background warm-start thread) ──────────
_vector_store  = None
_rag_engine    = None
_pipeline_ready = False   # True once warm-start completes


_init_lock = threading.Lock()

def _initialize_components():
    """
    Loads heavy ML components (CrossEncoder, ChromaDB, etc.).
    Called lazily on the first request to avoid Gunicorn fork() worker issues.
    """
    global _vector_store, _rag_engine, _pipeline_ready
    
    with _init_lock:
        if _pipeline_ready:
            return  # Avoid double initialization if concurrent requests arrive
            
        logger.info("Lazy-loading ML components...")
        try:
            from vectordb.chroma_store import ChromaVectorStore
            from rag_engine.engine import RAGEngine

            _vector_store = ChromaVectorStore()
            # This is the slow step — downloads/loads the CrossEncoder once
            _rag_engine   = RAGEngine(vector_store=_vector_store)
            _pipeline_ready = True
            logger.success("ML components successfully loaded.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")


def _run_pipeline(job_id: str, query: str, background: str):
    """Runs the full Skill Gap pipeline in a background thread."""
    try:
        if not _pipeline_ready:
            logger.info(f"[Job {job_id[:8]}] First run detected. Initializing pipeline...")
            _initialize_components()
            
        if not _pipeline_ready or _rag_engine is None or _vector_store is None:
            raise RuntimeError("Pipeline failed to initialize.")

        from scraper.registry import ScraperRegistry, check_freshness
        from ingestion.pipeline import IngestionPipeline
        from utils.cloud_storage import CloudStorage

        # 1. Freshness check
        cached = check_freshness(query)
        if cached:
            logger.info(f"[Job {job_id[:8]}] Cache HIT.")
            with _jobs_lock:
                _jobs[job_id] = {"status": "done", "result": cached}
            return

        # 2. Scrape
        logger.info(f"[Job {job_id[:8]}] Scraping...")
        registry = ScraperRegistry()
        all_jobs = registry.run_all(query=query)
        jobs_dicts = [asdict(j) for j in all_jobs]

        if not jobs_dicts:
            raise RuntimeError("No jobs found across all scrapers for this query.")

        # 3. Upload raw jobs to Supabase Storage
        try:
            storage = CloudStorage()
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            safe_query = query.replace(" ", "_")[:40]
            storage.upload_jobs(jobs_dicts, f"jobs/{date_str}_{safe_query}.json")
        except Exception as e:
            logger.warning(f"[Job {job_id[:8]}] Supabase upload skipped: {e}")

        # 4. Ingest — reset collection then re-embed
        _vector_store.reset_collection()
        pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)
        chunks = pipeline.run(jobs_dicts)
        _vector_store.add_documents(chunks)

        # 5. RAG analyze using pre-loaded CrossEncoder (fast)
        result = _rag_engine.analyze(query=query, user_background=background or None)

        # 6. Build output payload
        output_data = {
            "id":               str(uuid.uuid4()),
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "query":            result.query,
            "matched_jobs":     result.matched_jobs,
            "in_demand_skills": result.in_demand_skills,
            "analysis":         result.raw_analysis,
        }

        # 7. Persist to Supabase Postgres
        try:
            storage = CloudStorage()
            storage.insert_rag_output(output_data)
        except Exception as e:
            logger.warning(f"[Job {job_id[:8]}] Supabase insert skipped: {e}")

        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": output_data}
        logger.success(f"[Job {job_id[:8]}] ✓ Complete.")

    except Exception as e:
        logger.error(f"[Job {job_id[:8]}] ✗ Failed: {e}")
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route('/health')
def health():
    """Health check endpoint — shows warm-start status."""
    return jsonify({
        "status":  "ready" if _pipeline_ready else "warming_up",
        "message": "Pipeline ready." if _pipeline_ready else "ML components loading in background (~60s)."
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():

    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query field"}), 400

    query      = data['query']
    background = data.get('background', '')
    job_id     = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}

    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, query, background),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.route('/api/status/<job_id>', methods=['GET'])
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


if __name__ == '__main__':
    print("🚀 Starting SkillGap AI Server on http://127.0.0.1:8000")
    app.run(port=8000, debug=False)
