# viXTTS CLONE endpoint — RIENG voi endpoint giong (vi transformers <5 xung dot)
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# cryptography do OS cai, khong go duoc -> cai lai bang pip truoc
RUN pip install --no-cache-dir --ignore-installed cryptography

# Coqui TTS 0.27.5 (import "TTS") + transformers <5 (khac endpoint giong) + phu tro
RUN pip install --no-cache-dir \
        "coqui-tts==0.27.5" \
        "transformers>=4.57,<5.0" \
        "huggingface-hub>=0.34,<1.0" \
        librosa soundfile num2words runpod noisereduce

# Tai model viXTTS (~1.9GB) luc build -> cold-start nhanh
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('capleaf/viXTTS', local_dir='/models/viXTTS')"

COPY handler.py /handler.py
COPY refs /refs

CMD ["python", "-u", "/handler.py"]
