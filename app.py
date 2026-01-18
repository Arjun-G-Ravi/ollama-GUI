import os
import json
import psutil
import shutil
import uuid
import requests
import subprocess
import time
import base64
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file
from datetime import datetime
from io import BytesIO

app = Flask(__name__)

# --- Configuration ---
DEFAULT_MODEL = "devstral-small-2:latest"  # Model to preload on startup
OLLAMA_API_BASE = "http://localhost:11434/api"
PROMPTS_FILE = "prompts.json"
HISTORY_FILE = "history.json"
FAVORITES_FILE = "favorites.json"
MODEL_CARDS_FILE = "model_cards.json"
GENERATED_MEDIA_DIR = "generated_media"

# Ensure generated media directory exists
os.makedirs(GENERATED_MEDIA_DIR, exist_ok=True)

# HuggingFace inference tracking
hf_generation_status = {}  # Track ongoing generations

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
        if os.name == 'nt':
            os.system("taskkill /F /IM ollama.exe") # windows
        else:
            os.system("pkill ollama")
        time.sleep(1)
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
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
        except: pass

    return jsonify({
        "cpu": cpu_usage, "ram_text": ram_str,
        "gpu": gpu_usage, "vram_text": vram_str
    })

@app.route('/api/default_model')
def get_default_model():
    return jsonify({"model": DEFAULT_MODEL})

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

@app.route('/api/custom_cards')
def get_custom_cards():
    return jsonify(load_json(MODEL_CARDS_FILE, {}))

@app.route('/api/favorites', methods=['GET', 'POST'])
def handle_favorites():
    favorites = load_json(FAVORITES_FILE, [])
    if request.method == 'POST':
        name = request.json.get('name')
        if name in favorites: favorites.remove(name)
        else: favorites.append(name)
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
        if name in prompts: del prompts[name]
        save_json(PROMPTS_FILE, prompts)
        return jsonify({"status": "deleted"})
    return jsonify(prompts)

@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def handle_history():
    history = load_json(HISTORY_FILE, [])
    if request.method == 'POST':
        data = request.json
        chat_id = data.get('id')
        existing_index = next((i for (i, d) in enumerate(history) if d["id"] == chat_id), None)
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
                    if line: yield line + b'\n'
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode() + b'\n'
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/api/preload', methods=['POST'])
def preload_model():
    """Preload a model into VRAM by sending a minimal chat request"""
    try:
        model_name = request.json.get('model')
        # Send a minimal request to load the model
        requests.post(
            f"{OLLAMA_API_BASE}/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False
            },
            timeout=30
        )
        return jsonify({"status": "preloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- HuggingFace Integration ---

def run_hf_image_generation(task_id, model_id, prompt, params):
    """Run HuggingFace image generation in background thread"""
    try:
        hf_generation_status[task_id] = {"status": "loading", "progress": 0, "message": "Loading model..."}
        
        from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline
        import torch
        
        hf_generation_status[task_id] = {"status": "loading", "progress": 10, "message": "Initializing pipeline..."}
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load the appropriate pipeline
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        except:
            # Fallback to standard SD pipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
            except:
                pass
        
        hf_generation_status[task_id] = {"status": "generating", "progress": 30, "message": "Generating image..."}
        
        # Extract generation parameters
        width = params.get("width", 512)
        height = params.get("height", 512)
        num_inference_steps = params.get("steps", 30)
        guidance_scale = params.get("guidance_scale", 7.5)
        negative_prompt = params.get("negative_prompt", "")
        seed = params.get("seed", -1)
        
        # Set up generator for reproducibility
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        image = result.images[0]
        
        # Save image
        filename = f"{task_id}.png"
        filepath = os.path.join(GENERATED_MEDIA_DIR, filename)
        image.save(filepath)
        
        # Convert to base64 for preview
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        hf_generation_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Generation complete!",
            "image_base64": img_base64,
            "filepath": filepath
        }
        
        # Clean up
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        hf_generation_status[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }


def run_hf_video_generation(task_id, model_id, prompt, params):
    """Run HuggingFace video generation in background thread"""
    try:
        hf_generation_status[task_id] = {"status": "loading", "progress": 0, "message": "Loading video model..."}
        
        import torch
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        hf_generation_status[task_id] = {"status": "loading", "progress": 10, "message": "Initializing video pipeline..."}
        
        # Try to import video generation libraries
        try:
            from diffusers import DiffusionPipeline, TextToVideoSDPipeline
            pipe = TextToVideoSDPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        except:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        
        pipe = pipe.to(device)
        
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
            except:
                pass
        
        hf_generation_status[task_id] = {"status": "generating", "progress": 30, "message": "Generating video frames..."}
        
        # Extract generation parameters
        num_frames = params.get("num_frames", 16)
        fps = params.get("fps", 8)
        width = params.get("width", 512)
        height = params.get("height", 512)
        num_inference_steps = params.get("steps", 25)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed", -1)
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate video frames
        result = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        frames = result.frames[0] if hasattr(result, 'frames') else result.images
        
        hf_generation_status[task_id] = {"status": "encoding", "progress": 80, "message": "Encoding video..."}
        
        # Save as video file
        filename = f"{task_id}.mp4"
        filepath = os.path.join(GENERATED_MEDIA_DIR, filename)
        
        # Use imageio to save video
        import imageio
        import numpy as np
        
        # Convert frames to numpy arrays if needed
        frame_arrays = []
        for frame in frames:
            if hasattr(frame, 'numpy'):
                frame_arrays.append(frame.numpy())
            elif hasattr(frame, '__array__'):
                frame_arrays.append(np.array(frame))
            else:
                frame_arrays.append(frame)
        
        imageio.mimwrite(filepath, frame_arrays, fps=fps)
        
        hf_generation_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Video generation complete!",
            "filepath": filepath,
            "filename": filename
        }
        
        # Clean up
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        hf_generation_status[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }


@app.route('/api/hf/generate_image', methods=['POST'])
def hf_generate_image():
    """Start HuggingFace image generation"""
    try:
        data = request.json
        model_id = data.get('model')
        prompt = data.get('prompt')
        params = data.get('params', {})
        
        if not model_id or not prompt:
            return jsonify({"error": "Model and prompt are required"}), 400
        
        task_id = str(uuid.uuid4())
        
        # Start generation in background thread
        thread = threading.Thread(
            target=run_hf_image_generation,
            args=(task_id, model_id, prompt, params)
        )
        thread.start()
        
        return jsonify({"task_id": task_id, "status": "started"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/hf/generate_video', methods=['POST'])
def hf_generate_video():
    """Start HuggingFace video generation"""
    try:
        data = request.json
        model_id = data.get('model')
        prompt = data.get('prompt')
        params = data.get('params', {})
        
        if not model_id or not prompt:
            return jsonify({"error": "Model and prompt are required"}), 400
        
        task_id = str(uuid.uuid4())
        
        # Start generation in background thread
        thread = threading.Thread(
            target=run_hf_video_generation,
            args=(task_id, model_id, prompt, params)
        )
        thread.start()
        
        return jsonify({"task_id": task_id, "status": "started"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/hf/status/<task_id>')
def hf_generation_status_check(task_id):
    """Check status of HuggingFace generation task"""
    if task_id in hf_generation_status:
        return jsonify(hf_generation_status[task_id])
    return jsonify({"status": "not_found"}), 404


@app.route('/api/hf/download/<task_id>')
def hf_download_media(task_id):
    """Download generated media file"""
    if task_id in hf_generation_status:
        status = hf_generation_status[task_id]
        if status.get("status") == "completed" and "filepath" in status:
            return send_file(status["filepath"], as_attachment=True)
    return jsonify({"error": "File not found"}), 404


@app.route('/api/hf/models')
def get_hf_models():
    """Get list of HuggingFace models from model_cards.json"""
    cards = load_json(MODEL_CARDS_FILE, {})
    hf_models = {k: v for k, v in cards.items() if v.get("backend") == "huggingface"}
    return jsonify(hf_models)

if __name__ == '__main__':
    load_json(PROMPTS_FILE, {"Default": "You are a helpful assistant."})
    load_json(HISTORY_FILE, [])
    load_json(FAVORITES_FILE, [])
    load_json(MODEL_CARDS_FILE, {})
    
    # Preload default model
    def preload_default_model():
        # time.sleep(2)  # Wait for Flask to start
        try:
            print(f"Preloading model: {DEFAULT_MODEL}")
            requests.post(
                f"{OLLAMA_API_BASE}/chat",
                json={
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False
                },
                timeout=30
            )
            print(f"Model {DEFAULT_MODEL} preloaded successfully")
        except Exception as e:
            print(f"Failed to preload model: {e}")
    
    threading.Thread(target=preload_default_model, daemon=True).start()
    app.run(port=5000, debug=True)