# VieNeu-TTS MCP server

Exposes VieNeu-TTS to other tool agents (OMP / Oh My Pi, Claude Code, Codex, …)
via the [Model Context Protocol](https://modelcontextprotocol.io). Agents call
tools instead of the Python API.

Source: [`apps/mcp_server.py`](../apps/mcp_server.py). It wraps the in-process
`vieneu` SDK (same API as `apps/web_stream.py`).

## Tools

| Tool | Args | Returns |
|---|---|---|
| `list_backends` | – | available backends + current selection |
| `set_backend` | `backend`, `precision="fp32"`, `onnx_dir=None` | reloads model on next synth |
| `list_voices` | – | `[{id, name}, …]` preset voices for current model |
| `synthesize` | `text`, `voice_id=None`, `out_path=None`, `temperature=0.4`, `top_k=50` | `{path, sample_rate, num_samples, duration_sec, …}` — writes a 48 kHz mono WAV and returns its path |

## Install

```bash
uv sync                      # or: pip install -e .
uv add 'mcp[cli]'            # or: pip install 'vieneu[mcp]'
```

## Run

```bash
# stdio (default) — how agent clients launch it
uv run python -m apps.mcp_server

# HTTP/SSE — for remote agents
VIENEU_MCP_TRANSPORT=sse VIENEU_MCP_HOST=0.0.0.0 VIENEU_MCP_PORT=8010 \
    uv run python -m apps.mcp_server
```

Model selection via env (defaults = ONNX/CPU int8, lightest, matches web_stream):
`VIENEU_BACKEND` (auto|onnx|standard|lmdeploy), `VIENEU_PRECISION`,
`VIENEU_ONNX_DIR`, `VIENEU_OUTPUT_DIR` (where WAVs are written).

## Register with an agent

Create `.mcp.json` in the project (OMP reads `mcpServers`; Claude Code / Codex
use the same shape). Replace the path with your checkout.

```json
{
  "mcpServers": {
    "vieneu-tts": {
      "command": "uv",
      "args": ["run", "python", "-m", "apps.mcp_server"],
      "cwd": "/home/leesun/VieNeu-TTS",
      "env": { "VIENEU_BACKEND": "onnx", "VIENEU_PRECISION": "fp32" }
    }
  }
}
```

Then in OMP: `omp` picks it up automatically (project MCP config is enabled via
`mcp.enableProjectConfig=true`). The agent can now call `vieneu-tts.synthesize`,
`vieneu-tts.list_voices`, etc.

## Smoke test (no agent needed)

```bash
# lists tools + calls list_voices via the MCP dev inspector
uv run mcp dev apps/mcp_server.py
```

> First `synthesize` call downloads/loads the model (heavy) and needs the
> `vieneu` runtime deps installed; test on a host with the model available.
