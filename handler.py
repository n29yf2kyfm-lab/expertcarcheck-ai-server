"""
================================================================================
AI MECHANIC v10.0 — ULTIMATE
Everything in one handler:
  1. AI Car Mechanic (diagnosis, voice, OBD)
  2. AI 3D Car Generator (photo → 3D model)
  3. RAG Knowledge Base (teachable)
  4. AI Video (repair guides)
  
GPU: RTX A6000 48GB (fits everything)
Idle: $0 | Active: $0.49/hr
Version: 10.0.0-ultimate
================================================================================
"""

import os, sys, io, json, base64, tempfile, time, requests, hashlib, re
import torch, numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List

# ── FIX: TripoSR float→int resize bug ────────────────────────────────────
# TripoSR does image.resize(w / scale, h / scale) which crashes on newer PIL
# Monkey-patch to force int dimensions on all resize calls
_orig_resize = Image.Image.resize
def _safe_resize(self, size, *args, **kwargs):
    if isinstance(size, tuple):
        size = (int(size[0]), int(size[1]))
    return _orig_resize(self, size, *args, **kwargs)
Image.Image.resize = _safe_resize
from functools import lru_cache

# ── BOOT ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
VRAM_GB = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0
USE_INT4 = VRAM_GB < 40
GPU_COST_HR = 0.49

print(f"[BOOT] AI MECHANIC ULTIMATE v10.0 | {GPU_NAME} | {VRAM_GB}GB | INT4={USE_INT4}",
      file=sys.stderr, flush=True)

total_gpu_seconds = 0
total_requests = 0
cache_hits = 0
api_hits = 0
gpu_hits = 0
_models: Dict[str, Any] = {}

# ── CACHE ──────────────────────────────────────────────────────────────────
_response_cache: Dict[str, Dict] = {}
_tts_cache: Dict[str, Dict] = {}
CACHE_MAX_SIZE = 2000

def _cache_key(endpoint_type, **kwargs):
    clean = {k: v for k, v in kwargs.items() if k not in ("job_id", "timestamp", "request_id")}
    return f"{endpoint_type}:{hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:16]}"

def _get_cached(key: str, cache_dict=None, ttl=3600):
    global cache_hits
    cd = cache_dict or _response_cache
    entry = cd.get(key)
    if entry:
        if ttl == -1 or time.time() - entry["ts"] < ttl:
            cache_hits += 1
            entry["data"]["cached"] = True
            return entry["data"]
    return None

def _set_cached(key: str, data: Dict, cache_dict=None):
    cd = cache_dict or _response_cache
    if len(cd) >= CACHE_MAX_SIZE:
        oldest = min(cd, key=lambda k: cd[k]["ts"])
        del cd[oldest]
    cd[key] = {"data": data, "ts": time.time()}

# ── OBD DATABASE ───────────────────────────────────────────────────────────
from obd_database import DTC_DB, REPAIR_GUIDE

# ── RAG KNOWLEDGE BASE ─────────────────────────────────────────────────────
_knowledge_base: List[Dict[str, Any]] = []
_kb_initialized = False

def _init_knowledge_base():
    global _kb_initialized, _knowledge_base
    if _kb_initialized: return
    built_in = [
        {"id": "brake_pad_replace", "text": "To replace brake pads: 1) Loosen lug nuts 2) Jack up car 3) Remove wheel 4) Remove caliper bolts 5) Slide out old pads 6) Compress caliper piston 7) Install new pads 8) Reinstall caliper 9) Replace wheel 10) Torque lug nuts. Safety: Always use jack stands!", "source": "built-in", "topic": "brakes"},
        {"id": "oil_change", "text": "Oil change: 1) Warm engine 2) Park on level ground 3) Place drain pan 4) Remove drain plug 5) Let oil drain 6) Replace drain plug with new washer 7) Remove old oil filter 8) Lubricate new filter gasket 9) Install new filter 10) Add correct oil amount 11) Check with dipstick.", "source": "built-in", "topic": "maintenance"},
        {"id": "spark_plug_replace", "text": "Spark plug replacement: 1) Remove engine cover 2) Disconnect coil packs 3) Remove old plugs 4) Check gap on new plugs (0.028-0.044 inch) 5) Apply anti-seize 6) Install and torque to spec 7) Reconnect coils.", "source": "built-in", "topic": "ignition"},
        {"id": "misfire_diagnosis", "text": "Engine misfire: 1) Scan for codes P0300-P0308 2) Swap ignition coils to see if misfire moves 3) Inspect spark plugs 4) Test fuel injectors 5) Compression test 6) Check vacuum leaks with smoke test.", "source": "built-in", "topic": "diagnosis"},
        {"id": "overheating", "text": "Engine overheating: 1) Pull over safely 2) Turn heater to max 3) If temp keeps rising, shut off engine 4) Wait 30+ min before opening hood 5) Check coolant level 6) Check for leaks 7) Check thermostat 8) Check radiator fans 9) Check water pump. NEVER open hot radiator cap!", "source": "built-in", "topic": "emergency"},
        {"id": "check_engine_light", "text": "Check engine light: Solid yellow = fault detected, drive cautiously. Flashing = misfire detected STOP driving immediately. Common causes: loose gas cap, O2 sensor, MAF sensor, catalytic converter.", "source": "built-in", "topic": "diagnosis"},
        {"id": "timing_belt", "text": "Timing belt replacement: Typically every 60k-100k miles. If belt breaks on interference engine = catastrophic damage. Signs: ticking noise, engine misfire, oil leak from front of motor.", "source": "built-in", "topic": "engine"},
        {"id": "dpf_cleaning", "text": "DPF cleaning: Diesel Particulate Filter. If clogged: 1) Drive at highway speed for 30 min (passive regen) 2) If light still on, dealer forced regeneration needed 3) Never ignore DPF light.", "source": "built-in", "topic": "emissions"},
        {"id": "clutch_replacement", "text": "Clutch replacement signs: slipping under acceleration, high biting point, difficulty changing gear. Typical life: 50k-100k miles depending on driving style.", "source": "built-in", "topic": "transmission"},
        {"id": "turbo_failure", "text": "Turbo failure signs: whining noise, excessive smoke, loss of power, check engine light. Causes: oil starvation, foreign object damage, worn bearings. Replacement cost: £800-£2500.", "source": "built-in", "topic": "engine"},
        {"id": "egr_valve", "text": "EGR valve cleaning: Exhaust Gas Recirculation. Clogged EGR causes: rough idle, poor acceleration, engine stall. Clean with carb cleaner or replace. Common on diesels.", "source": "built-in", "topic": "emissions"},
        {"id": "suspension_check", "text": "Suspension check: 1) Bounce each corner - should settle in 1-2 bounces 2) Check for leaking shock absorbers 3) Listen for knocking over bumps (worn bushes) 4) Check tyre wear patterns.", "source": "built-in", "topic": "suspension"},
        {"id": "battery_test", "text": "Car battery: Typical life 3-5 years. Test with multimeter: 12.6V+ = fully charged, 12.4V = 75%, 12.2V = 50%, 12.0V = 25%. Below 11.8V = replace immediately.", "source": "built-in", "topic": "electrical"},
        {"id": "alternator_check", "text": "Alternator check: With engine running, battery voltage should be 13.5V-14.5V. If lower: alternator failing. Signs: dim headlights, battery warning light, electrical issues.", "source": "built-in", "topic": "electrical"},
        {"id": "brake_disc_skimming", "text": "Brake disc skimming: If discs are warped (pulsing pedal) or scored. Minimum thickness must be maintained. Cost: £30-£50 per disc. Replacement: £80-£200 per disc.", "source": "built-in", "topic": "brakes"},
    ]
    _knowledge_base = built_in
    _kb_initialized = True

_init_knowledge_base()

def _search_knowledge(query: str, top_k: int = 3) -> List[Dict]:
    query_words = set(query.lower().split())
    scored = []
    for entry in _knowledge_base:
        score = 0
        entry_text = entry["text"].lower()
        for word in query_words:
            if word in entry_text: score += 1
            if word in entry.get("topic", "").lower(): score += 2
            if word in entry.get("id", "").lower(): score += 2
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]

# ── MODEL LOADERS (lazy) ───────────────────────────────────────────────────

def get_deepseek():
    if "deepseek" not in _models:
        q = "int4" if USE_INT4 else "fp8"
        print(f"[LOAD] DeepSeek-R1 ({q})...", file=sys.stderr, flush=True)
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                      tensor_parallel_size=1, gpu_memory_utilization=0.5,
                      max_model_len=8192, quantization=q, dtype="float16", trust_remote_code=True)
            _models["deepseek"] = (llm, SamplingParams)
            print("[LOAD] DeepSeek ready", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[LOAD] DeepSeek failed: {e}", file=sys.stderr, flush=True)
            _models["deepseek"] = None
    return _models["deepseek"]

def get_gemma():
    if "gemma" not in _models:
        print("[LOAD] Gemma-E4B...", file=sys.stderr, flush=True)
        try:
            from transformers import pipeline
            pipe = pipeline("image-text-to-text", model="google/gemma-3n-E4B-it",
                            device=0 if torch.cuda.is_available() else -1, torch_dtype=torch.bfloat16)
            _models["gemma"] = pipe
            print("[LOAD] Gemma ready", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[LOAD] Gemma failed: {e}", file=sys.stderr, flush=True)
            _models["gemma"] = None
    return _models["gemma"]

def get_kokoro():
    if "kokoro" not in _models:
        print("[LOAD] Kokoro...", file=sys.stderr, flush=True)
        try:
            from kokoro import KModel, KPipeline
            model = KModel().to(DEVICE).eval()
            pipeline = KPipeline(lang_code='a')
            _models["kokoro"] = {"model": model, "pipeline": pipeline}
            print("[LOAD] Kokoro ready", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[LOAD] Kokoro failed: {e}", file=sys.stderr, flush=True)
            _models["kokoro"] = None
    return _models["kokoro"]

def get_parakeet():
    if "parakeet" not in _models:
        print("[LOAD] Parakeet...", file=sys.stderr, flush=True)
        try:
            from transformers import pipeline
            pipe = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3",
                            device=0 if torch.cuda.is_available() else -1, torch_dtype=torch.float16)
            _models["parakeet"] = pipe
            print("[LOAD] Parakeet ready", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[LOAD] Parakeet failed: {e}", file=sys.stderr, flush=True)
            _models["parakeet"] = None
    return _models["parakeet"]

def get_tripo_sr():
    """TripoSR removed — use MiDaS depth fallback instead."""
    return None

def get_midas():
    if "midas" not in _models:
        print("[LOAD] MiDaS...", file=sys.stderr, flush=True)
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to(DEVICE).eval()
            _models["midas"] = midas
            _models["midas_transform"] = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            print("[LOAD] MiDaS ready", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[LOAD] MiDaS failed: {e}", file=sys.stderr, flush=True)
            _models["midas"] = None
    return _models["midas"], _models.get("midas_transform")

# ── API OVERFLOW ───────────────────────────────────────────────────────────

def _call_api(msgs, max_tok=512, temp=0.7):
    global api_hits
    # Groq
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": msgs, "max_tokens": max_tok, "temperature": temp},
                timeout=15)
            if r.status_code == 200:
                api_hits += 1
                return {"response": r.json()["choices"][0]["message"]["content"], "backend": "groq", "cost": "$0", "gpu_used": False}
        except: pass
    # OpenRouter
    or_key = os.environ.get("Openrouter_api_key", "")
    if or_key:
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {or_key}", "Content-Type": "application/json",
                         "HTTP-Referer": "https://expertcarcheck.com", "X-Title": "AI Mechanic"},
                json={"model": "openrouter/auto", "messages": msgs, "max_tokens": max_tok, "temperature": temp}, timeout=15)
            if r.status_code == 200:
                api_hits += 1
                return {"response": r.json()["choices"][0]["message"]["content"], "backend": "openrouter", "cost": "$0", "gpu_used": False}
        except: pass
    return None

# ── COST TRACKER ───────────────────────────────────────────────────────────

def _track(start):
    global total_gpu_seconds, gpu_hits
    elapsed = time.time() - start
    total_gpu_seconds += elapsed
    gpu_hits += 1
    return {"elapsed": round(elapsed, 2), "cost_usd": round(elapsed * GPU_COST_HR / 3600, 6), "gpu": GPU_NAME, "gpu_used": True}

# ── INFERENCE ──────────────────────────────────────────────────────────────

def _fmt_llama(msgs):
    out = ""
    for m in msgs:
        r, c = m.get("role", "user"), m.get("content", "")
        if r == "system": out += f"<|start_header_id|>system<|end_header_id|>\n{c}<|eot_id|>"
        elif r == "user": out += f"<|start_header_id|>user<|end_header_id|>\n{c}<|eot_id|>"
        elif r == "assistant": out += f"<|start_header_id|>assistant<|end_header_id|>\n{c}<|eot_id|>"
    out += "<|start_header_id|>assistant<|end_header_id|>\n"
    return out

# ── HANDLER: Chat ──────────────────────────────────────────────────────────

def handle_chat(inp):
    global total_requests
    total_requests += 1
    msgs = inp.get("messages", [])
    if isinstance(msgs, str): msgs = [{"role": "user", "content": msgs}]
    
    # Check cache
    cache_key = _cache_key("chat", messages=msgs)
    cached = _get_cached(cache_key, ttl=86400)
    if cached: cached["cost"] = "$0 (cache)"; return cached
    
    # OBD code
    last = str(msgs[-1].get("content", "")).lower() if msgs else ""
    if re.match(r"^\s*[pbcu]\d{4}\s*$", last, re.I):
        code = last.strip().upper()
        info = DTC_DB.get(code)
        if info:
            rep = REPAIR_GUIDE.get(code, {})
            result = {"response": f"**{code}**: {info['d']}\nSeverity: {info['s']}\nSystem: {info['sys']}\n" +
                      (f"Causes: {', '.join(rep.get('causes', []))}\n" if rep.get('causes') else "") +
                      (f"Parts: {', '.join(rep.get('parts', []))}" if rep.get('parts') else ""),
                      "model": "obd_db", "cost": "$0", "gpu_used": False}
            _set_cached(cache_key, result); return result
    
    # RAG
    kb = _search_knowledge(last, 3) if inp.get("use_rag", True) else []
    if kb:
        ctx = "\n\n".join([f"[{i+1}] {r['text'][:500]}" for i, r in enumerate(kb)])
        rag_msgs = msgs.copy()
        rag_msgs.insert(0, {"role": "system", "content": f"Expert car mechanic knowledge:\n\n{ctx}\n\nProvide practical step-by-step advice."})
        # Try API with RAG
        if inp.get("use_overflow", True):
            overflow = _call_api(rag_msgs)
            if overflow: overflow["rag"] = True; _set_cached(cache_key, overflow); return overflow
        # GPU with RAG
        data = get_deepseek()
        if data:
            llm, SP = data; start = time.time()
            if not any(m.get("role") == "system" for m in rag_msgs):
                rag_msgs.insert(0, {"role": "system", "content": "Expert automotive mechanic."})
            out = llm.generate(_fmt_llama(rag_msgs), SP(temperature=0.7, top_p=0.9, max_tokens=512))
            r = out[0].outputs[0].text.strip()
            if "</think>" in r: r = r.split("</think>")[-1].strip()
            result = {"response": r, "model": "deepseek-r1", "rag": True, **_track(start)}
            _set_cached(cache_key, result); return result
    
    # API overflow
    if inp.get("use_overflow", True):
        overflow = _call_api(msgs)
        if overflow: _set_cached(cache_key, overflow); return overflow
    
    # GPU fallback
    data = get_deepseek()
    if data:
        llm, SP = data; start = time.time()
        if not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": "Expert automotive mechanic AI."})
        out = llm.generate(_fmt_llama(msgs), SP(temperature=0.7, top_p=0.9, max_tokens=512))
        r = out[0].outputs[0].text.strip()
        if "</think>" in r: r = r.split("</think>")[-1].strip()
        result = {"response": r, "model": "deepseek-r1", **_track(start)}
        _set_cached(cache_key, result); return result
    
    return {"error": "All backends failed", "gpu_used": False}

# ── HANDLER: Speak ─────────────────────────────────────────────────────────

def handle_speak(inp):
    text = inp.get("text", "")
    if not text: return {"error": "'text' required", "gpu_used": False}
    voice = inp.get("voice", "af_bella")
    speed = inp.get("speed", 1.0)
    cache_key = _cache_key("tts", text=text, voice=voice)
    cached = _get_cached(cache_key, cache_dict=_tts_cache, ttl=-1)
    if cached: cached["cost"] = "$0 (tts cache)"; return cached
    
    k = get_kokoro()
    if k is None: return {"error": "Kokoro not loaded", "gpu_used": False}
    start = time.time()
    segs = [audio for _, _, audio in k["pipeline"](text, voice=voice, speed=speed, model=k["model"])]
    if not segs: return {"error": "No audio", "gpu_used": False}
    full = np.concatenate(segs)
    import scipy.io.wavfile as wavfile
    buf = io.BytesIO()
    wavfile.write(buf, 24000, (full * 32767).astype(np.int16) if full.dtype != np.int16 else full)
    result = {"audio_base64": base64.b64encode(buf.getvalue()).decode(), "format": "wav",
              "sr": 24000, "duration_ms": round(len(full) / 24), "engine": "kokoro", **_track(start)}
    _set_cached(cache_key, result, cache_dict=_tts_cache)
    return result

# ── HANDLER: Transcribe ────────────────────────────────────────────────────

def handle_transcribe(inp):
    audio_b64 = inp.get("audio_base64", "")
    if not audio_b64: return {"error": "'audio_base64' required", "gpu_used": False}
    audio = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio); tmp = f.name
    try:
        start = time.time()
        pipe = get_parakeet()
        if pipe is None: return {"error": "Parakeet not loaded", "gpu_used": False}
        r = pipe(tmp)
        return {"text": r.get("text", ""), "model": "parakeet", **_track(start)}
    finally:
        os.unlink(tmp)

# ── HANDLER: Vision ────────────────────────────────────────────────────────

def handle_vision(inp):
    img_b64 = inp.get("image_base64", "")
    prompt = inp.get("prompt", "What car part is this and what condition?")
    if not img_b64: return {"error": "'image_base64' required", "gpu_used": False}
    pipe = get_gemma()
    if pipe is None: return {"error": "Gemma not loaded", "gpu_used": False}
    start = time.time()
    try:
        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        r = pipe(text=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img}]}], max_new_tokens=512)
        text = r[0]["generated_text"][-1]["content"] if isinstance(r, list) else str(r)
        return {"response": text, "model": "gemma-e4b", **_track(start)}
    except Exception as e:
        return {"error": str(e), "gpu_used": False}

# ── HANDLER: 3D Generation ─────────────────────────────────────────────────

def handle_generate_3d(inp):
    """Generate 3D car model from image using TripoSR."""
    image_b64 = inp.get("image_base64", "")
    image_url = inp.get("image_url", "")
    quality = inp.get("quality", "high")
    
    # Load image
    image = None
    if image_b64:
        try: image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as e: return {"success": False, "error": f"Invalid image: {e}"}
    elif image_url:
        try:
            r = requests.get(image_url, timeout=30)
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e: return {"success": False, "error": f"Download failed: {e}"}
    else:
        return {"success": False, "error": "Provide image_base64 or image_url"}
    
    sizes = {"low": 256, "medium": 512, "high": 1024}
    target = sizes.get(quality, 512)
    image = image.resize((target, target), Image.LANCZOS)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name); img_path = f.name
    
    try:
        # 3D via MiDaS depth estimation
        return _generate_depth_fallback(image, quality)
    except Exception as e:
        import traceback
        print(f"[ERROR] 3D gen: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"success": False, "error": str(e)}
    finally:
        os.unlink(img_path)

def _generate_depth_fallback(image, quality):
    midas, transform = get_midas()
    if midas is None: return {"success": False, "error": "No 3D model available"}
    start = time.time()
    input_batch = transform(np.array(image)).to(DEVICE)
    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False).squeeze()
    depth = pred.cpu().numpy()
    h, w = depth.shape
    verts, faces = [], []
    scale = 0.01
    for y in range(h - 1):
        for x in range(w - 1):
            i = len(verts)
            verts.extend([
                [x*scale - w*scale/2, y*scale - h*scale/2, depth[y,x]*scale],
                [(x+1)*scale - w*scale/2, y*scale - h*scale/2, depth[y,x+1]*scale],
                [x*scale - w*scale/2, (y+1)*scale - h*scale/2, depth[y+1,x]*scale],
                [(x+1)*scale - w*scale/2, (y+1)*scale - h*scale/2, depth[y+1,x+1]*scale]
            ])
            faces.extend([[i,i+1,i+2], [i+1,i+3,i+2]])
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.remove_duplicate_vertices(); mesh.remove_degenerate_faces()
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        glb_path = f.name
    mesh.export(glb_path)
    with open(glb_path, 'rb') as f:
        glb_data = base64.b64encode(f.read()).decode()
    os.unlink(glb_path)
    t = _track(start)
    return {"success": True, "model_base64": glb_data, "format": "glb",
            "method": "MiDaS_depth", "quality": quality, "vertices": len(mesh.vertices),
            "faces": len(mesh.faces), "elapsed": t["elapsed"], "note": "Fallback - lower fidelity"}

# ── HANDLER: OBD ───────────────────────────────────────────────────────────

def handle_obd(inp):
    action = inp.get("action", "lookup")
    if action == "lookup":
        code = inp.get("code", "").upper().strip()
        cache_key = _cache_key("obd", code=code)
        cached = _get_cached(cache_key, ttl=-1)
        if cached: cached["cost"] = "$0"; return cached
        info = DTC_DB.get(code)
        if not info: return {"code": code, "found": False, "cost": "$0", "gpu_used": False}
        rep = REPAIR_GUIDE.get(code, {})
        result = {"code": code, "found": True, "desc": info["d"], "severity": info["s"], "system": info["sys"],
                  "causes": rep.get("causes", []), "parts": rep.get("parts", []), "cost": "$0", "gpu_used": False}
        _set_cached(cache_key, result); return result
    elif action == "batch":
        return {"results": [{"code": c, **DTC_DB.get(c, {})} for c in [x.upper().strip() for x in inp.get("codes", [])]], "cost": "$0", "gpu_used": False}
    elif action == "list":
        return {"total": len(DTC_DB), "codes": {k: v["d"] for k, v in DTC_DB.items()}, "cost": "$0", "gpu_used": False}
    elif action == "diagnose":
        dtcs = [c.upper().strip() for c in inp.get("dtcs", [])]
        results = [DTC_DB.get(c) for c in dtcs if DTC_DB.get(c)]
        sev = sum({"low": 1, "moderate": 2, "severe": 3}.get(r["s"], 1) for r in results)
        overall = "critical" if sev >= 7 else "warning" if sev >= 4 else "advisory"
        return {"diagnosis": overall, "codes": results, "cost": "$0", "gpu_used": False}
    return {"error": f"Unknown action: {action}", "gpu_used": False}

# ── HANDLER: Knowledge ─────────────────────────────────────────────────────

def handle_knowledge(inp):
    action = inp.get("action", "search")
    if action == "add":
        _knowledge_base.append({"id": inp.get("doc_id", f"doc_{int(time.time())}"), "text": inp.get("text", ""), "topic": inp.get("topic", "general"), "source": "user"})
        return {"added": True, "total": len(_knowledge_base), "cost": "$0", "gpu_used": False}
    elif action == "search":
        return {"results": _search_knowledge(inp.get("query", ""), inp.get("top_k", 3)), "cost": "$0", "gpu_used": False}
    elif action == "list":
        return {"entries": [{"id": e["id"], "topic": e.get("topic", ""), "preview": e["text"][:80]} for e in _knowledge_base], "cost": "$0", "gpu_used": False}
    elif action == "summary":
        return {"total": len(_knowledge_base), "cost": "$0", "gpu_used": False}
    return {"error": f"Unknown: {action}", "gpu_used": False}

# ── HANDLER: Status ────────────────────────────────────────────────────────

def handle_status(inp):
    return {
        "status": "healthy", "gpu": GPU_NAME, "vram_gb": VRAM_GB, "int4": USE_INT4,
        "cost_hr": GPU_COST_HR, "loaded": list(_models.keys()),
        "stats": {"total": total_requests, "gpu": gpu_hits, "api": api_hits, "cache": cache_hits, "gpu_sec": round(total_gpu_seconds, 1)},
        "models": [
            {"name": "DeepSeek-R1-32B", "role": "Car diagnosis", "loaded": "deepseek" in _models},
            {"name": "Gemma-3N-E4B", "role": "Vision/Vibration", "loaded": "gemma" in _models},
            {"name": "Kokoro-82M", "role": "Text-to-speech", "loaded": "kokoro" in _models},
            {"name": "Parakeet-TDT", "role": "Speech-to-text", "loaded": "parakeet" in _models},
            {"name": "TripoSR", "role": "3D generation", "loaded": "tripo_sr" in _models},
            {"name": "MiDaS", "role": "Depth estimation", "loaded": "midas" in _models},
            {"name": "OBD Database", "role": "200+ fault codes", "loaded": True},
            {"name": "RAG Knowledge", "role": "Teachable docs", "entries": len(_knowledge_base)},
        ],
        "version": "10.0.0-ultimate",
    }

# ── MAIN HANDLER ───────────────────────────────────────────────────────────

def handler(job):
    inp = job.get("input", {})
    if "input" in inp: inp = inp["input"]
    jt = inp.get("type", "chat").lower()
    jid = job.get("id", "?")
    print(f"[JOB] {jt} | {jid}", file=sys.stderr, flush=True)
    try:
        handlers = {
            "chat": handle_chat, "speak": handle_speak, "tts": handle_speak,
            "transcribe": handle_transcribe, "stt": handle_transcribe,
            "vision": handle_vision, "see": handle_vision,
            "obd": handle_obd, "diagnose": handle_obd,
            "generate_3d": handle_generate_3d, "3d": handle_generate_3d, "image_to_3d": handle_generate_3d,
            "knowledge": handle_knowledge, "teach": handle_knowledge, "rag": handle_knowledge,
            "status": handle_status,
        }
        result = handlers.get(jt, lambda x: {"error": f"Unknown: '{jt}'", "available": list(handlers.keys()), "gpu_used": False})(inp)
        if isinstance(result, dict):
            result["_meta"] = {"v": "10.0.0", "jid": jid, "models_loaded": list(_models.keys())}
        return result
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": str(e), "gpu_used": False, "_meta": {"v": "10.0.0", "jid": jid}}

if __name__ == "__main__":
    print("=" * 60, file=sys.stderr)
    print("AI MECHANIC ULTIMATE v10.0", file=sys.stderr)
    print(f"GPU: {GPU_NAME} | {VRAM_GB}GB | INT4={USE_INT4}", file=sys.stderr)
    print("Models: DeepSeek + Gemma + Kokoro + Parakeet + TripoSR + MiDaS", file=sys.stderr)
    print("Features: Chat + Voice + Vision + OBD + 3D Gen + RAG + Teach", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    import runpod
    runpod.serverless.start({"handler": handler})
