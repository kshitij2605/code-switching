#!/usr/bin/env bash
# Launch the GGUF code-switch ASR server (llama.cpp liquid-audio) in a docker container.
#
# The prebuilt runner needs glibc 2.38; the GPU host has 2.34, so we run it inside ubuntu:24.04 (CPU).
# It serves an OpenAI-compatible, streaming /v1/chat/completions endpoint. Point the app at it with
#   serve_api.py --asr-gguf-url http://localhost:${HOST_PORT}
# See memory: kaigi-gguf-asr-runner.
set -euo pipefail

GGUF_ROOT="${GGUF_ROOT:-/data/lfm2.5/run-a/gguf}"
RUNNER="${RUNNER:-$GGUF_ROOT/liquid_runner}"
BACKBONE="${BACKBONE:-$GGUF_ROOT/output/gguf_f32_step2500/lfm25-audio-jp-step2500-backbone-f32.gguf}"
MMPROJ="${MMPROJ:-$GGUF_ROOT/work/gguf_bf16/mmproj-lfm25-audio-jp-encoder-bf16.gguf}"
HOST_PORT="${HOST_PORT:-8090}"
NAME="${NAME:-kaigi-gguf-asr}"
IMAGE="${IMAGE:-ubuntu:24.04}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-8}"

# Mount the filesystem root that holds both the runner and the GGUFs (so absolute paths resolve in-container).
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "${HOST_PORT}:8080" \
  -v "$RUNNER:/r:ro" -v /data:/data:ro \
  -e LD_LIBRARY_PATH=/r \
  "$IMAGE" /r/llama-liquid-audio-server \
  -m "$BACKBONE" -mm "$MMPROJ" \
  --host 0.0.0.0 --port 8080 -c "$CTX" -t "$THREADS" >/dev/null

echo "Started '$NAME' -> http://localhost:${HOST_PORT}  (backbone=$(basename "$BACKBONE"))"
echo "Waiting for model load ..."
for _ in $(seq 1 60); do
  if docker logs "$NAME" 2>&1 | grep -q "Server ready"; then
    echo "READY. Point the app at it:  python scripts/serve_api.py --asr-gguf-url http://localhost:${HOST_PORT}"
    exit 0
  fi
  sleep 2
done
echo "WARN: server not ready after 120s; check: docker logs $NAME"
exit 1
