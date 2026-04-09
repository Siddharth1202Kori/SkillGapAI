import sys
import subprocess
import json
import os

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='frontend')

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query field"}), 400
        
    query = data['query']
    background = data.get('background', '')
    
    # sys.executable ensures Render always uses the correct venv Python,
    # not a system-level `python` that may not exist or point to wrong version.
    cmd = [sys.executable, "-u", "main.py", "--query", query, "--reset-db"]
    if background.strip():
        cmd.extend(["--user-background", background.strip()])
        
    print(f"Running pipeline with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        out_path = os.path.join('rag_outputs', 'base_version', 'output_analysis.json')
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                analysis_data = json.load(f)
            return jsonify(analysis_data)
        else:
            return jsonify({"error": "Pipeline finished but output file not found."}), 500
            
    except subprocess.CalledProcessError as e:
        print("PIPELINE ERROR:")
        print(e.stderr)
        print(e.stdout)
        return jsonify({"error": "Pipeline failed to execute. Check Render logs."}), 500

if __name__ == '__main__':
    print("🚀 Starting SkillGap AI Server on http://127.0.0.1:8000")
    app.run(port=8000, debug=True)
