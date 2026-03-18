#!/usr/bin/env bash

echo "Worker Initiated"

echo "Starting WebUI API"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
/workspace/stable-diffusion-webui/venv/bin/python /workspace/stable-diffusion-webui/webui.py \
  --xformers \
  --skip-python-version-check \
  --skip-torch-cuda-test \
  --skip-install \
  --ckpt /workspace/stable-diffusion-webui/models/Stable-diffusion/realvisxlV50_v40Bakedvae.safetensors \
  --disable-safe-unpickle \
  --port 7860 \
  --api \
  --nowebui \
  --skip-version-check \
  --opt-channelslast \
  --upcast-sampling \
  --no-hashing \
  --cuda-malloc \
  --no-download-sd-model &

echo "Starting RunPod Handler"
/workspace/stable-diffusion-webui/venv/bin/python -u /workspace/handler.py
