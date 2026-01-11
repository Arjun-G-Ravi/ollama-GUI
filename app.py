import os
import json
import time
import psutil
import requests
import subprocess
import shutil
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
OLLAMA_API_BASE = "http://localhost:11434/api"
PROMPTS_FILE = "prompts.json"
HISTORY_FILE = "history.json"

# --- Data Management Helpers ---
def load_json(filename, default):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump(default, f)
        return default
    with open(filename, "r") as f:
        return json.load(f)

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    # CPU & RAM
    cpu_usage = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    ram_used_gb = round(ram.used / (1024**3), 1)
    ram_total_gb = round(ram.total / (1024**3), 1)

    # GPU (NVIDIA only via nvidia-smi)
    gpu_usage = 0
    vram_usage = 0
    vram_used_gb = 0
    vram_total_gb = 0
    
    if shutil.which('nvidia-smi'):
        try:
            # Get GPU Load
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'], 
                encoding='utf-8'
            )
            data = result.strip().split(',')
            gpu_usage = int(data[0])
            vram_used_mb = int(data[1])
            vram_total_mb = int(data[2])
            vram_usage = round((vram_used_mb / vram_total_mb) * 100, 1)
            vram_used_gb = round(vram_used_mb / 1024, 1)
            vram_total_gb = round(vram_total_mb / 1024, 1)
        except:
            pass # Fail silently if nvidia-smi fails

    return jsonify({
        "cpu": cpu_usage,
        "ram_percent": ram_usage,
        "ram_text": f"{ram_used_gb}/{ram_total_gb} GB",
        "gpu": gpu_usage,
        "vram_percent": vram_usage,
        "vram_text": f"{vram_used_gb}/{vram_total_gb} GB"
    })

@app.route('/api/models')
def get_models():
    try:
        resp = requests.get(f"{OLLAMA_API_BASE}/tags")
        return jsonify(resp.json())
    except:
        return jsonify({"models": []})

# --- Prompt Management ---
@app.route('/api/prompts', methods=['GET', 'POST', 'DELETE'])
def handle_prompts():
    prompts = load_json(PROMPTS_FILE, {"Default": "You are a helpful assistant."})
    
    if request.method == 'POST':
        data = request.json
        prompts[data['name']] = data['content']
        save_json(PROMPTS_FILE, prompts)
        return jsonify({"status": "saved"})
    
    if request.method == 'DELETE':
        name = request.args.get('name')
        if name in prompts:
            del prompts[name]
            save_json(PROMPTS_FILE, prompts)
        return jsonify({"status": "deleted"})

    return jsonify(prompts)

# --- History Management ---
@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def handle_history():
    history = load_json(HISTORY_FILE, [])
    
    if request.method == 'POST':
        # Save a completed conversation
        data = request.json
        entry = {
            "id": int(time.time()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": data.get("model"),
            "messages": data.get("messages")
        }
        history.insert(0, entry) # Prepend
        save_json(HISTORY_FILE, history[:50]) # Keep last 50
        return jsonify({"status": "saved"})

    if request.method == 'DELETE':
        # Clear all history
        save_json(HISTORY_FILE, [])
        return jsonify({"status": "cleared"})

    return jsonify(history)

# --- Chat Streaming ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model')
    messages = data.get('messages') # List of {role, content}

    def generate():
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        try:
            with requests.post(f"{OLLAMA_API_BASE}/chat", json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        yield line + b'\n'
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode() + b'\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    # Create files if missing
    load_json(PROMPTS_FILE, {"Default": "You are a helpful assistant."})
    load_json(HISTORY_FILE, [])
    app.run(port=5000, debug=True)