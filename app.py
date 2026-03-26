from flask import Flask, request, jsonify, send_from_directory
import subprocess
import json
import os

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
    
    # We use subprocess to run the actual pipeline script so it isolates the run.
    cmd = ["python", "-u", "main.py", "--query", query, "--reset-db"]
    # We do NOT use --no-supabase so it uploads to your storage bucket as intended
    if background.strip():
        cmd.extend(["--user-background", background.strip()])
        
    print(f"Running pipeline with command: {' '.join(cmd)}")
    try:
        # Run process (this may take 30-60 secs depending on scraping + mistral)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # main.py saves the final result to output_analysis.json
        if os.path.exists('output_analysis.json'):
            with open('output_analysis.json', 'r') as f:
                analysis_data = json.load(f)
            return jsonify(analysis_data)
        else:
            return jsonify({"error": "Pipeline finished but failed to generate output_analysis.json"}), 500
            
    except subprocess.CalledProcessError as e:
        print("PIPELINE ERROR:")
        print(e.stderr)
        print(e.stdout)
        return jsonify({"error": "Pipeline failed to execute. Check terminal logs."}), 500

if __name__ == '__main__':
    print("🚀 Starting SkillGap AI Server on http://127.0.0.1:8000")
    app.run(port=8000, debug=True)
