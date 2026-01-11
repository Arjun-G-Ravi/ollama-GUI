import os
import json
import time
import psutil
import subprocess
import shutil
import uuid
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
OLLAMA_API_BASE = "http://localhost:11434/api"
PROMPTS_FILE = "prompts.json"
HISTORY_FILE = "history.json"

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

@app.route('/api/kill', methods=['POST'])
def kill_ollama():
    print('\ncow')
    return jsonify({"status": "killed", "message": "Ollama process terminated."})
    # try:
    #     # Works on Linux/Mac. For Windows use 'taskkill /F /IM ollama.exe'
    #     os.system("pkill ollama") 
    #     return jsonify({"status": "killed", "message": "Ollama process terminated."})
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def get_stats():
    # CPU & RAM
    cpu_usage = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    
    # Format: 7.6/16GB
    ram_used = round(ram.used / (1024**3), 1)
    ram_total = round(ram.total / (1024**3), 0) # Integer for total usually looks cleaner
    ram_str = f"{ram_used}/{int(ram_total)}GB"

    # GPU (NVIDIA)
    gpu_usage = 0
    vram_str = "N/A"
    
    if shutil.which('nvidia-smi'):
        try:
            # Fast query
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'], 
                encoding='utf-8'
            )
            data = result.strip().split(',')
            gpu_usage = int(data[0])
            vram_used = int(data[1])
            vram_total = int(data[2])
            
            # Convert MB to GB
            vram_used_gb = round(vram_used / 1024, 1)
            vram_total_gb = round(vram_total / 1024, 0)
            vram_str = f"{vram_used_gb}/{int(vram_total_gb)}GB"
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
        resp = requests.get(f"{OLLAMA_API_BASE}/tags")
        return jsonify(resp.json())
    except:
        return jsonify({"models": []})

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

@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def handle_history():
    history = load_json(HISTORY_FILE, [])
    
    if request.method == 'POST':
        data = request.json
        chat_id = data.get('id')
        
        # Check if updating existing conversation
        existing_index = next((index for (index, d) in enumerate(history) if d["id"] == chat_id), None)
        
        entry = {
            "id": chat_id if chat_id else str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": data.get("model"),
            "messages": data.get("messages")
        }

        if existing_index is not None:
            history[existing_index] = entry # Update
            # Move to top
            history.insert(0, history.pop(existing_index))
        else:
            history.insert(0, entry) # Create new at top
            
        save_json(HISTORY_FILE, history[:50]) # Limit to 50
        return jsonify({"status": "saved", "id": entry["id"]})

    if request.method == 'DELETE':
        save_json(HISTORY_FILE, [])
        return jsonify({"status": "cleared"})

    return jsonify(history)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model')
    messages = data.get('messages')

    def generate():
        import requests # Import here to avoid global scope issues in threads
        payload = { "model": model, "messages": messages, "stream": True }
        try:
            with requests.post(f"{OLLAMA_API_BASE}/chat", json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        yield line + b'\n'
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode() + b'\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    load_json(PROMPTS_FILE, {"Default": "You are a helpful assistant."})
    load_json(HISTORY_FILE, [])
    app.run(port=5000, debug=True)