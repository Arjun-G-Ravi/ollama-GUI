import os
import json
import psutil
import shutil
import uuid
import requests
import subprocess
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
OLLAMA_API_BASE = "http://localhost:11434/api"
PROMPTS_FILE = "prompts.json"
HISTORY_FILE = "history.json"
FAVORITES_FILE = "favorites.json"
MODEL_CARDS_FILE = "model_cards.json" # New file for templates

# --- Helpers ---
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

@app.route('/api/health')
def check_health():
    try:
        requests.get(f"{OLLAMA_API_BASE}/tags", timeout=0.5)
        return jsonify({"status": "online"})
    except:
        return jsonify({"status": "offline"})

@app.route('/api/restart', methods=['POST'])
def restart_server():
    try:
        if os.name == 'nt': # Windows
            os.system("taskkill /F /IM ollama.exe")
        else: # Linux/Mac
            os.system("pkill ollama")
        
        time.sleep(1)
        
        subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return jsonify({"status": "restarting"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    cpu_usage = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_str = f"{round(ram.used / (1024**3), 1)} / {int(round(ram.total / (1024**3), 0))} GB"

    gpu_usage = 0
    vram_str = "N/A"
    
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'], 
                encoding='utf-8'
            )
            lines = result.strip().split('\n')
            if lines:
                data = lines[0].split(',')
                gpu_usage = int(data[0].strip())
                vram_used_gb = round(int(data[1].strip()) / 1024, 1)
                vram_total_gb = int(round(int(data[2].strip()) / 1024, 0))
                vram_str = f"{vram_used_gb} / {vram_total_gb} GB"
        except:
            pass

    return jsonify({
        "cpu": cpu_usage,
        "ram_text": ram_str,
        "gpu": gpu_usage,
        "vram_text": vram_str
    })

@app.route('/api/models')
def get_models():
    try:
        resp = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=2)
        return jsonify(resp.json())
    except:
        return jsonify({"models": [], "error": "Ollama is offline"})

@app.route('/api/model_details', methods=['POST'])
def get_model_details():
    try:
        resp = requests.post(f"{OLLAMA_API_BASE}/show", json={"name": request.json.get('name')}, timeout=2)
        return jsonify(resp.json())
    except:
        return jsonify({"details": {}})

# --- New Route for Custom Cards ---
@app.route('/api/custom_cards')
def get_custom_cards():
    return jsonify(load_json(MODEL_CARDS_FILE, {}))

@app.route('/api/favorites', methods=['GET', 'POST'])
def handle_favorites():
    favorites = load_json(FAVORITES_FILE, [])
    if request.method == 'POST':
        name = request.json.get('name')
        if name in favorites:
            favorites.remove(name)
        else:
            favorites.append(name)
        save_json(FAVORITES_FILE, favorites)
    return jsonify(favorites)

@app.route('/api/prompts', methods=['GET', 'POST', 'DELETE'])
def handle_prompts():
    prompts = load_json(PROMPTS_FILE, {"Default": "You are a helpful assistant."})
    if request.method == 'POST':
        prompts[request.json['name']] = request.json['content']
        save_json(PROMPTS_FILE, prompts)
        return jsonify({"status": "saved"})
    if request.method == 'DELETE':
        name = request.args.get('name')
        if name in prompts:
            del prompts[name]
            save_json(PROMPTS_FILE, prompts)
        return jsonify({"status": "deleted"})
    return jsonify(prompts)

@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def handle_history():
    history = load_json(HISTORY_FILE, [])
    
    if request.method == 'POST':
        data = request.json
        chat_id = data.get('id')
        
        existing_index = next((index for (index, d) in enumerate(history) if d["id"] == chat_id), None)
        
        entry = {
            "id": chat_id if chat_id else str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": data.get("model"),
            "messages": data.get("messages")
        }

        if existing_index is not None:
            history[existing_index] = entry 
            history.insert(0, history.pop(existing_index))
        else:
            history.insert(0, entry)
            
        save_json(HISTORY_FILE, history[:50])
        return jsonify({"status": "saved", "id": entry["id"]})

    if request.method == 'DELETE':
        save_json(HISTORY_FILE, [])
        return jsonify({"status": "cleared"})

    return jsonify(history)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    
    def generate():
        try:
            with requests.post(f"{OLLAMA_API_BASE}/chat", json={**data, "stream": True}, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        yield line + b'\n'
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode() + b'\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    load_json(PROMPTS_FILE, {"Default": "You are a helpful assistant."})
    load_json(HISTORY_FILE, [])
    load_json(FAVORITES_FILE, [])
    load_json(MODEL_CARDS_FILE, {}) # Ensure custom cards file exists
    app.run(port=5000, debug=True)