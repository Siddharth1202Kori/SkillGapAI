import sys
import subprocess
import json
import os
import uuid
import threading

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='frontend')

# ── In-memory job store ─────────────────────────────────────────────────────
# Stores job_id -> { status, result, error }
# Reset on every Render restart (acceptable since jobs are short-lived)
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _run_pipeline(job_id: str, query: str, background: str):
    """
    Background thread that runs the full Skill Gap pipeline.
    Stores the result in `_jobs` when done.
    """
    try:
        cmd = [sys.executable, "-u", "main.py", "--query", query, "--reset-db"]
        if background.strip():
            cmd.extend(["--user-background", background.strip()])

        print(f"[Job {job_id[:8]}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])  # Last 2000 chars of stderr

        out_path = os.path.join('rag_outputs', 'base_version', 'output_analysis.json')
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                data = json.load(f)
            with _jobs_lock:
                _jobs[job_id] = {"status": "done", "result": data}
            print(f"[Job {job_id[:8]}] ✓ Complete")
        else:
            raise FileNotFoundError("Pipeline finished but output file not found.")

    except Exception as e:
        print(f"[Job {job_id[:8]}] ✗ Failed: {e}")
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Non-blocking endpoint — launches the pipeline in a background thread
    and immediately returns a job_id. Frontend polls /api/status/<job_id>.
    """
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query field"}), 400

    query = data['query']
    background = data.get('background', '')
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}

    # Fire the heavy pipeline in a daemon thread — won't block gunicorn
    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, query, background),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id, "status": "running"}), 202


@app.route('/api/status/<job_id>', methods=['GET'])
def job_status(job_id: str):
    """
    Polling endpoint: frontend calls this every 3s until status == 'done' or 'error'.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Unknown job_id"}), 404

    return jsonify(job)


if __name__ == '__main__':
    print("🚀 Starting SkillGap AI Server on http://127.0.0.1:8000")
    app.run(port=8000, debug=True)
