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

# ============ CHUAN HOA VAN BAN (so + viet tat + phat am) ============
_ONES = ["không","một","hai","ba","bốn","năm","sáu","bảy","tám","chín"]
_PRON = {"Seven":"Sê Vờn","MiniMax":"Mi Ni Mách","YouTube":"Diu Túp","Youtube":"Diu Túp",
         "Facebook":"Phây Búc","Zalo":"Za Lô","TikTok":"Tích Tóc","Google":"Gu Gồ"}
_LETTER = {'A':'a','Ă':'á','Â':'ớ','B':'bê','C':'xê','D':'đê','Đ':'đê','E':'e','Ê':'ê','F':'ép',
           'G':'gờ','H':'hát','I':'i','J':'di','K':'ca','L':'lờ','M':'mờ','N':'nờ','O':'o','Ô':'ô',
           'Ơ':'ơ','P':'pê','Q':'quy','R':'rờ','S':'ét','T':'tê','U':'u','Ư':'ư','V':'vê','W':'vê kép',
           'X':'ích','Y':'i','Z':'dét'}
_SYM = [('%',' phần trăm'),('$',' đô la'),('&',' và '),('@',' a còng '),('=',' bằng '),('°',' độ '),('+',' cộng ')]

def _read3(n, leading):
    tr, ch, dv = n//100, (n%100)//10, n%10
    out = []
    if tr > 0: out.append(_ONES[tr]+" trăm")
    elif not leading and (ch>0 or dv>0): out.append("không trăm")
    if ch > 1:
        out.append(_ONES[ch]+" mươi")
        if dv==1: out.append("mốt")
        elif dv==5: out.append("lăm")
        elif dv>0: out.append(_ONES[dv])
    elif ch == 1:
        out.append("mười")
        if dv==5: out.append("lăm")
        elif dv>0: out.append(_ONES[dv])
    else:
        if dv>0:
            if tr>0 or not leading: out.append("lẻ")
            out.append(_ONES[dv])
    return " ".join(out)

def _num_to_vi(num):
    num = int(num)
    if num == 0: return "không"
    neg = num < 0; num = abs(num)
    groups = []
    while num > 0:
        groups.append(num%1000); num//=1000
    units = ["","nghìn","triệu","tỷ","nghìn tỷ","triệu tỷ"]
    parts, n = [], len(groups)
    for i in range(n-1, -1, -1):
        if groups[i]==0 and i>0: continue
        seg = _read3(groups[i], i==n-1)
        if seg: parts.append(seg + ((" "+units[i]) if units[i] else ""))
    r = re.sub(r"\s+"," "," ".join(parts)).strip()
    return ("âm "+r) if neg else r

def _num_repl_vi(m):
    s = m.group(0)
    if "," in s:
        intp, dec = s.replace(".","").split(",",1)
        return _num_to_vi(intp or "0")+" phẩy "+" ".join(_ONES[int(d)] for d in dec if d.isdigit())
    return _num_to_vi(s.replace(".",""))

def _acr_repl(m):
    s = m.group(0)
    if re.search(r"[AĂÂEÊIOÔƠUƯY]", s):   # co nguyen am -> viet hoa nhan manh, doc thuong
        return s.lower()
    return " ".join(_LETTER.get(c,c) for c in s)   # toan phu am -> viet tat, danh van

def _apply_pron(text):
    for w in sorted(_PRON.keys(), key=len, reverse=True):
        text = re.sub(r"\b"+re.escape(w)+r"\b", _PRON[w], text, flags=re.IGNORECASE)
    return text

_NUM2W = {"en":"en","es":"es","e":"es","fr":"fr","f":"fr","it":"it","i":"it","pt":"pt","p":"pt",
          "de":"de","ru":"ru","nl":"nl","pl":"pl","tr":"tr","cs":"cz","hu":"hu"}
_DEC_COMMA = {"es","e","fr","f","it","i","pt","p","de","ru","nl","pl","tr","cs","hu"}

def normalize_text(text, lang):
    lang = (lang or "vi").lower()
    if lang == "vi":
        text = _apply_pron(text)
        text = re.sub(r"(\d[\d.,]*)\s*[đ₫]", lambda m: m.group(1)+" đồng", text)
        for a,b in _SYM: text = text.replace(a,b)
        text = re.sub(r"\b[A-ZĐÂĂÊÔƠƯ]{2,}\b", _acr_repl, text)
        text = re.sub(r"\d[\d.,]*\d|\d", _num_repl_vi, text)
        return re.sub(r"\s+"," ",text).strip()
    code = _NUM2W.get(lang)
    if not code: return text
    try:
        from num2words import num2words
    except Exception:
        return text
    def _repl(m):
        s = m.group(0)
        try:
            s2 = s.replace(".","").replace(",",".") if lang in _DEC_COMMA else s.replace(",","")
            num = float(s2) if "." in s2 else int(s2)
            return num2words(num, lang=code)
        except Exception:
            return s
    return re.sub(r"\d[\d.,]*\d|\d", _repl, text)

def _denoise_ref(ref_path):
    """Loc tap am mau giong (noisereduce) -> clone ro va giong hon."""
    try:
        import noisereduce as nr, librosa
        y, sr = librosa.load(ref_path, sr=None, mono=True)
        yd = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.85)
        out = ref_path + ".dn.wav"
        sf.write(out, yd, sr)
        return out
    except Exception:
        return ref_path

def _is_glitch(wav, sr, text):
    """Phat hien chunk loi: qua dai (ren) / qua ngan (mat tieng) / im lang dai o giua (stall)."""
    try:
        n = len(wav)
        if n < int(sr * 0.15): return True
        dur = n / float(sr)
        nt = len(re.sub(r"\s", "", text or ""))
        if nt < 2: return False
        exp = nt / 13.0
        if dur > exp * 2.3 + 0.8: return True
        if dur < exp * 0.32: return True
        aw = np.abs(wav[:int(n * 0.92)])
        quiet = (aw < 0.02).astype(np.int8)
        if quiet.any():
            dd = np.diff(np.concatenate(([0], quiet, [0])))
            st = np.where(dd == 1)[0]; en = np.where(dd == -1)[0]
            if len(st) and int((en - st).max()) / float(sr) > 0.75:
                return True
        return False
    except Exception:
        return False

def _clone_infer(m, ch, lang, gpt_cond, spk, kw):
    try:
        o = m.inference(ch, lang, gpt_cond, spk, **kw)
        return np.asarray(o["wav"], dtype="float32")
    except Exception:
        return None

def split_text(text, max_chars=180):
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
        # do bieu cam 0..1 -> temperature 0.30..0.90 (ghi de preset neu co)
        expr = inp.get("expressiveness")
        temp = (0.30 + 0.60*max(0.0, min(1.0, float(expr)))) if (expr is not None and expr != "") else preset["temperature"]
        # loc tap am mau giong (tuy chon) -> clone ro hon
        if inp.get("denoise"):
            ref_path = _denoise_ref(ref_path)
        gpt_cond, spk = m.get_conditioning_latents(
            audio_path=ref_path, gpt_cond_len=30, gpt_cond_chunk_len=6, max_ref_length=60)
        wavs = []
        text = normalize_text(text, lang)   # so->chu tieng Viet/nuoc ngoai, viet tat, phat am
        _kw = dict(temperature=temp, repetition_penalty=preset.get("repetition_penalty", 10.0),
                   length_penalty=1.0, top_k=30, top_p=0.85, speed=spd, enable_text_splitting=False)
        _retried = 0
        for ch in split_text(text):
            # AUTO-RETRY: tao lai chunk bi loi (ren/lang/mat tieng) -> giao ban SACH
            w = _clone_infer(m, ch, lang, gpt_cond, spk, _kw)
            _t = 0
            while (w is None or _is_glitch(w, SR, ch)) and _t < 3:
                _w2 = _clone_infer(m, ch, lang, gpt_cond, spk, _kw)
                if _w2 is not None and len(_w2) > 0: w = _w2
                _t += 1
            if _t: _retried += 1
            if w is None or len(w) < 1: w = np.zeros(int(SR * 0.1), dtype="float32")
            wavs.append(w)
        audio = _concat(wavs)
        data, fmt = _encode(audio, want)
        return {
            "audio_base64": base64.b64encode(data).decode(),
            "format": fmt, "chars": len(text),
            "audio_seconds": round(len(audio) / SR, 2),
            "gen_seconds": round(time.time() - t0, 2),
            "retried_chunks": _retried,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-1500:]}

runpod.serverless.start({"handler": handler})
