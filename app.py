from flask import Flask, render_template, request, jsonify
import requests
import json
import os

app = Flask(__name__)

OLLAMA_API_BASE = "http://localhost:11434/api"
PROMPTS_FILE = "prompts.json"

# Ensure prompts file exists
if not os.path.exists(PROMPTS_FILE):
    with open(PROMPTS_FILE, "w") as f:
        json.dump({"Default": "You are a helpful AI assistant."}, f)

def get_saved_prompts():
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        return jsonify(response.json())
    except:
        return jsonify({"models": []})

@app.route('/api/prompts', methods=['GET', 'POST'])
def handle_prompts():
    prompts = get_saved_prompts()
    if request.method == 'POST':
        data = request.json
        prompts[data['name']] = data['content']
        with open(PROMPTS_FILE, "w") as f:
            json.dump(prompts, f)
        return jsonify({"status": "success"})
    return jsonify(prompts)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    model = data.get('model')
    prompt = data.get('prompt')
    system_prompt = data.get('system_prompt', "")

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_API_BASE}/generate", json=payload)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)