# -*- coding: utf-8 -*-
"""
TAN SEVEN VOICE - viXTTS CLONE handler (RunPod Serverless).
Nhan bam giong: bo mau tham chieu -> ra giong do (18 ngon ngu, gom tieng Viet).
Input JSON:
  { "text": "...", "language": "vi",
    "ref_b64": "<wav base64>"  hoac  "ref_name": "a_hung",
    "style": "📰 Tin tức", "speed": 1.0, "format": "mp3" }
Copy y het engine.py cua tool may (viXTTS + patch tieng Viet).
"""
import os, io, re, time, base64, tempfile, subprocess
import numpy as np
import soundfile as sf
import runpod

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "/models/viXTTS"
REF_DIR = "/refs"
SR = 24000

# ===== PATCH bat buoc cho viXTTS =====
import torch
torch.set_num_threads(max(1, os.cpu_count() or 1))
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

import re as _re
from TTS.tts.layers.xtts import tokenizer as _xtts_tok
_orig_pre = _xtts_tok.VoiceBpeTokenizer.preprocess_text
def _vi_pre(self, txt, lang):
    if lang.split("-")[0] == "vi":
        return _re.sub(r"\s+", " ", txt.lower()).strip()
    return _orig_pre(self, txt, lang)
_xtts_tok.VoiceBpeTokenizer.preprocess_text = _vi_pre

# ===== CHU DE / CAM XUC (temperature + speed + repetition_penalty) =====
STYLE_PRESETS = {
    "🤖 Tự động (cân bằng)":   {"temperature": 0.45, "speed": 1.00, "repetition_penalty": 10.0},
    "📰 Tin tức":              {"temperature": 0.35, "speed": 1.05, "repetition_penalty": 10.0},
    "📖 Kể chuyện":            {"temperature": 0.60, "speed": 0.98, "repetition_penalty": 8.0},
    "📚 Sách nói":             {"temperature": 0.45, "speed": 0.96, "repetition_penalty": 10.0},
    "🧘 Tâm linh / Thiền":     {"temperature": 0.40, "speed": 0.85, "repetition_penalty": 10.0},
    "✝️ Kinh thánh":           {"temperature": 0.40, "speed": 0.85, "repetition_penalty": 10.0},
    "🎬 Quảng cáo (sôi động)": {"temperature": 0.62, "speed": 1.10, "repetition_penalty": 8.0},
    "👶 Thiếu nhi (vui tươi)": {"temperature": 0.65, "speed": 1.02, "repetition_penalty": 8.0},
    "🎙️ Trang trọng / MC":     {"temperature": 0.45, "speed": 0.96, "repetition_penalty": 10.0},
    "🌙 Thơ / Ngâm":           {"temperature": 0.50, "speed": 0.82, "repetition_penalty": 10.0},
}
DEFAULT_STYLE = "🤖 Tự động (cân bằng)"

# ===== NAP MODEL (1 lan) =====
_model = None
def get_model():
    global _model
    if _model is None:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        cfg = XttsConfig()
        cfg.load_json(os.path.join(MODEL_DIR, "config.json"))
        m = Xtts.init_from_config(cfg)
        m.load_checkpoint(cfg, checkpoint_dir=MODEL_DIR, use_deepspeed=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(dev); m.eval()
        try: m.tokenizer.char_limits.setdefault("vi", 230)
        except Exception: pass
        _model = m
    return _model

def split_text(text, max_chars=220):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text: return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if not cur: cur = p
        elif len(cur) + len(p) + 1 <= max_chars: cur += " " + p
        else: chunks.append(cur); cur = p
    if cur: chunks.append(cur)
    return chunks or [text]

def _concat(wavs, gap=0.2):
    if not wavs: return np.zeros(1, dtype="float32")
    sil = np.zeros(int(SR * gap), dtype="float32")
    out = []
    for w in wavs:
        out.append(np.asarray(w, dtype="float32")); out.append(sil)
    return np.concatenate(out)

def _encode(audio, want="mp3"):
    if want == "mp3":
        with tempfile.TemporaryDirectory() as d:
            wp = os.path.join(d, "a.wav"); mp = os.path.join(d, "a.mp3")
            sf.write(wp, np.asarray(audio, dtype="float32"), SR)
            try:
                subprocess.run(["ffmpeg", "-y", "-i", wp, "-codec:a", "libmp3lame", "-q:a", "2", mp],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                with open(mp, "rb") as f: return f.read(), "mp3"
            except Exception: pass
    buf = io.BytesIO(); sf.write(buf, np.asarray(audio, dtype="float32"), SR, format="WAV")
    return buf.getvalue(), "wav"

def handler(job):
    inp = job.get("input") or {}
    if inp.get("list_refs"):
        try: return {"refs": [f[:-4] for f in os.listdir(REF_DIR) if f.endswith(".wav")]}
        except Exception as e: return {"error": str(e)}
    text = (inp.get("text") or "").strip()
    lang = (inp.get("language") or inp.get("lang") or "vi")
    style = inp.get("style") or DEFAULT_STYLE
    speed = inp.get("speed")
    want = (inp.get("format") or "mp3").lower()
    if not text: return {"error": "Thiếu 'text'"}

    # mau tham chieu: base64 hoac ten co san
    ref_path = None
    if inp.get("ref_b64"):
        try:
            data = base64.b64decode(inp["ref_b64"])
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.write(data); tf.close(); ref_path = tf.name
        except Exception as e:
            return {"error": "ref_b64 lỗi: %s" % e}
    elif inp.get("ref_name"):
        ref_path = os.path.join(REF_DIR, os.path.basename(inp["ref_name"]) + ".wav")
    if not ref_path or not os.path.exists(ref_path):
        return {"error": "Thiếu mẫu giọng (ref_b64 hoặc ref_name)"}

    t0 = time.time()
    try:
        m = get_model()
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])
        spd = float(speed) if speed else preset["speed"]
        gpt_cond, spk = m.get_conditioning_latents(
            audio_path=ref_path, gpt_cond_len=30, gpt_cond_chunk_len=6, max_ref_length=60)
        wavs = []
        for ch in split_text(text):
            out = m.inference(
                ch, lang, gpt_cond, spk,
                temperature=preset["temperature"],
                repetition_penalty=preset.get("repetition_penalty", 10.0),
                length_penalty=1.0, top_k=30, top_p=0.85,
                speed=spd, enable_text_splitting=False)
            wavs.append(np.asarray(out["wav"], dtype="float32"))
        audio = _concat(wavs)
        data, fmt = _encode(audio, want)
        return {
            "audio_base64": base64.b64encode(data).decode(),
            "format": fmt, "chars": len(text),
            "audio_seconds": round(len(audio) / SR, 2),
            "gen_seconds": round(time.time() - t0, 2),
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-1500:]}

runpod.serverless.start({"handler": handler})
