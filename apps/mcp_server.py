"""
VieNeu-TTS MCP server
=====================
Exposes VieNeu-TTS as MCP tools so other coding/tool agents (OMP, Claude Code,
Codex, etc.) can synthesize Vietnamese speech without touching the Python API.

Wraps the in-process `vieneu` SDK (the same public API used by apps/web_stream.py):
    Vieneu(backend=..., precision=...) -> .list_preset_voices(), .infer(text, voice=...)

Run (stdio, for agent clients):
    uv run python -m apps.mcp_server
Run (HTTP/SSE, for remote agents):
    VIENEU_MCP_TRANSPORT=sse VIENEU_MCP_HOST=0.0.0.0 VIENEU_MCP_PORT=8010 \
        uv run python -m apps.mcp_server

Model selection via env (defaults match web_stream: ONNX/CPU int8, lightest):
    VIENEU_BACKEND=onnx        # auto | onnx | standard | lmdeploy
    VIENEU_PRECISION=fp32      # backend-specific
    VIENEU_ONNX_DIR=/path      # optional local ONNX export
    VIENEU_OUTPUT_DIR=/path    # where synthesized .wav files are written
"""
from __future__ import annotations

import os
import io
import wave
import time
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    # mcp SDK 2.x: FastMCP was renamed to MCPServer.
    from mcp.server.mcpserver import MCPServer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'mcp' package (SDK >= 2.0) is required. Install it, e.g.:\n"
        "    uv add 'mcp[cli]'\n"
        "or:  pip install 'mcp[cli]'"
    ) from e

SAMPLE_RATE = 48_000

_BACKEND = os.environ.get("VIENEU_BACKEND", "onnx")
_PRECISION = os.environ.get("VIENEU_PRECISION", "fp32")
_ONNX_DIR = os.environ.get("VIENEU_ONNX_DIR") or None
_OUTPUT_DIR = Path(os.environ.get("VIENEU_OUTPUT_DIR", tempfile.gettempdir())) / "vieneu_mcp"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mcp = MCPServer("vieneu-tts")

# Lazily-constructed singleton — model load is heavy, so defer until first use.
_engine: Any = None
_engine_key: tuple[str, str, Optional[str]] = ("", "", None)


def _get_engine():
    """Return the current Vieneu engine, constructing/reloading it if config changed."""
    global _engine, _engine_key
    key = (_BACKEND, _PRECISION, _ONNX_DIR)
    if _engine is None or key != _engine_key:
        from vieneu import Vieneu  # imported lazily so `--help`/import stays cheap

        kwargs: dict[str, Any] = {"backend": _BACKEND, "precision": _PRECISION}
        if _ONNX_DIR:
            kwargs["onnx_dir"] = _ONNX_DIR
        _engine = Vieneu(**kwargs)
        _engine_key = key
    return _engine


def _to_wav_bytes(audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm16 = (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


@mcp.tool()
def list_backends() -> dict:
    """List available synthesis backends and the currently selected one.

    Backends: "auto" (pick best available), "onnx" (CPU int8, lightest),
    "standard" (PyTorch/transformers GPU), "lmdeploy" (fast GPU serving).
    """
    return {
        "available": ["auto", "onnx", "standard", "lmdeploy"],
        "current": {"backend": _BACKEND, "precision": _PRECISION, "onnx_dir": _ONNX_DIR},
    }


@mcp.tool()
def set_backend(backend: str, precision: str = "fp32", onnx_dir: Optional[str] = None) -> dict:
    """Switch the active backend/precision. Triggers a model reload on next synthesize.

    Args:
        backend: one of "auto", "onnx", "standard", "lmdeploy".
        precision: backend-specific precision tag (e.g. "fp32", "int8").
        onnx_dir: optional path to a local ONNX export.
    """
    global _BACKEND, _PRECISION, _ONNX_DIR, _engine
    if backend not in {"auto", "onnx", "standard", "lmdeploy"}:
        raise ValueError(f"Unknown backend '{backend}'. Use auto|onnx|standard|lmdeploy.")
    _BACKEND, _PRECISION, _ONNX_DIR = backend, precision, onnx_dir
    _engine = None  # force reload on next use
    return {"ok": True, "backend": _BACKEND, "precision": _PRECISION, "onnx_dir": _ONNX_DIR}


@mcp.tool()
def list_voices() -> list[dict]:
    """List preset voices available for the current model, as [{id, name}, ...]."""
    eng = _get_engine()
    voices = eng.list_preset_voices()
    out: list[dict] = []
    for item in voices:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            name, vid = item
            out.append({"id": vid, "name": name})
        else:
            out.append({"id": str(item), "name": str(item)})
    return out or [{"id": "", "name": "(no preset voices)"}]


@mcp.tool()
def synthesize(
    text: str,
    voice_id: Optional[str] = None,
    out_path: Optional[str] = None,
    temperature: float = 0.4,
    top_k: int = 50,
) -> dict:
    """Synthesize Vietnamese speech to a 48 kHz mono WAV file and return its path.

    Args:
        text: the Vietnamese text to speak.
        voice_id: preset voice id from list_voices(); None uses the model default.
        out_path: destination .wav path; defaults to a file under VIENEU_OUTPUT_DIR.
        temperature: sampling temperature.
        top_k: sampling top-k.

    Returns:
        {path, sample_rate, num_samples, duration_sec, voice_id, backend, elapsed_sec}
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    eng = _get_engine()
    t0 = time.perf_counter()
    audio = eng.infer(text, voice=voice_id or None, temperature=temperature, top_k=top_k)
    elapsed = time.perf_counter() - t0

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav_bytes = _to_wav_bytes(audio)

    dest = Path(out_path) if out_path else _OUTPUT_DIR / f"tts_{int(time.time()*1000)}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(wav_bytes)

    return {
        "path": str(dest),
        "sample_rate": SAMPLE_RATE,
        "num_samples": int(audio.shape[0]),
        "duration_sec": round(audio.shape[0] / SAMPLE_RATE, 3),
        "voice_id": voice_id,
        "backend": _BACKEND,
        "elapsed_sec": round(elapsed, 3),
    }


def main() -> None:
    transport = os.environ.get("VIENEU_MCP_TRANSPORT", "stdio").lower()
    if transport in ("sse", "http"):
        import asyncio

        host = os.environ.get("VIENEU_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("VIENEU_MCP_PORT", "8010"))
        asyncio.run(mcp.run_sse_async(host=host, port=port))
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
