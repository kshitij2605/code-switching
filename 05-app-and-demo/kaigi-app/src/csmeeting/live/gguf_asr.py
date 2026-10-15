"""GGUF ASR client — talks to the llama.cpp `llama-liquid-audio-server` (the code-switch ASR fine-tune as GGUF).

The server is OpenAI-compatible but **streaming-only**: POST /v1/chat/completions with a system prompt
("Perform ASR.") and a user `input_audio` part (base64 WAV), then accumulate the SSE deltas. One prompt
handles JA + EN (language-agnostic), so `lang` is accepted but ignored. Runs the GGUF on CPU (in a docker
container on this host — see scripts/gguf_asr_server.sh); see memory: kaigi-gguf-asr-runner.

Stdlib only (urllib/wave) so the live package gains no new deps. Corporate HTTP proxy is bypassed for the
(usually localhost) server URL.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
import wave

import numpy as np

_SPECIAL = re.compile(r"<\|[^|]*\|>")


class GgufAsrClient:
    def __init__(self, base_url: str, prompt: str = "Perform ASR.", max_tokens: int = 256, timeout: float = 180.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        # bypass any HTTP(S)_PROXY env — the server is local and corporate proxies 503 it
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _wav_bytes(self, wav: np.ndarray, sr: int) -> bytes:
        x = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        pcm = (x * 32767.0).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(pcm)
        return buf.getvalue()

    def transcribe(self, wav: np.ndarray, sr: int, lang: str | None = None) -> str:
        """Audio -> text via the GGUF server. `lang` is ignored ("Perform ASR." is language-agnostic)."""
        if wav is None or len(wav) == 0:
            return ""
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        b64 = base64.b64encode(self._wav_bytes(wav, sr)).decode("ascii")
        body = {
            "model": "", "stream": True, "max_tokens": self.max_tokens, "temperature": 0.0,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]},
            ],
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        out: list[str] = []
        with self._opener.open(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    out.append(json.loads(payload)["choices"][0]["delta"].get("content", "") or "")
                except Exception:  # noqa: BLE001 — skip keepalive / non-delta frames
                    pass
        return _SPECIAL.sub("", "".join(out)).strip()
