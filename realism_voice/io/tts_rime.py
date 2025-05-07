import os
import requests
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────
RIME_TTS_URL = os.getenv("RIME_API_BASE", "https://users.rime.ai/v1/rime-tts")
RIME_API_KEY = os.getenv("RIME_API_KEY")
DEFAULT_SPK  = os.getenv("RIME_SPEAKER", "orion")

# ─── Arcana Streaming‑MP3 ───────────────────────────────────────────────────
def rime_tts_stream(
    text: str,
    speaker: str = DEFAULT_SPK,
    output_path: Path = Path("remi_output.mp3"),
) -> Path:
    """
    Streams MP3 bytes from Rime Arcana and writes them to disk chunk-by-chunk.
    """
    headers = {
        "Accept":        "audio/mp3",               # trigger streaming MP3
        "Authorization": f"Bearer {RIME_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "speaker":           speaker,
        "text":              text,
        "modelId":           "arcana",               # Arcana for lifelike voice
        "audioFormat":       "mp3",                  # ensure MP3 payload
        "reduceLatency":     True,                   # shave off extra processing
        "samplingRate":      22050,                  # or lower for telephony
        "repetition_penalty":1.2,
        "temperature":       0.7,
        "top_p":             0.9,
        "max_tokens":        1200,
    }

    # stream=True ensures we start getting audio the instant the first bytes arrive
    resp = requests.post(RIME_TTS_URL, json=payload, headers=headers, stream=True)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)

    return output_path 