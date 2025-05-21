import base64
import os
import subprocess
import time
from typing import Dict, Any

import requests
import runpod

SD_WEBUI_URL = "http://127.0.0.1:7860"          # внутри контейнера

# ───────────────────────────────── helpers ────────────────────────────────────
def encode(b: bytes) -> str:
    return base64.b64encode(b).decode()

def start_webui() -> None:
    """
    Стартуем AUTOMATIC1111 в фоне ровно один раз при инициализации воркера.
    """
    if os.getenv("_WEBUI_STARTED"):
        return

    os.environ["_WEBUI_STARTED"] = "1"
    # фоновый запуск: всё логи выводятся в stdout контейнера
    subprocess.Popen(
        ["bash", "-c", ". venv/bin/activate && python launch.py ${COMMANDLINE_ARGS}"],
        cwd="/workspace/stable-diffusion-webui",
    )
    # ждём, пока REST-порт засветится
    for _ in range(60):
        try:
            if requests.get(f"{SD_WEBUI_URL}/docs", timeout=1).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    else:
        raise RuntimeError("Stable Diffusion WebUI did not start")

# ───────────────────────────────── handler ───────────────────────────────────
def handler(job: Dict[str, Any]):               # ← signature требование RunPod
    start_webui()                               # гарантируем, что SD поднят

    j = job["input"]                            # JSON из запроса
    try:
        payload = {
            "init_images": [j["init_image"]],
            "mask": j["mask_image"],
            "prompt": j.get("prompt", ""),
            "negative_prompt": j.get("negative_prompt", ""),
            "seed": j.get("seed", -1),
            "steps": j.get("steps", 30),
            "cfg_scale": j.get("cfg_scale", 7.0),
            "width": 1016,
            "height": 1016,
            "sampler_name": "Euler a",
            "scheduler": "SGM Uniform",
            "denoising_strength": 1,
            "override_settings": {
                "sd_model_checkpoint": "realvisxlV50_v40Bakedvae"
            },
            "alwayson_scripts": {
                "ControlNet": j["controlnet"],
            },
        }

        r = requests.post(f"{SD_WEBUI_URL}/sdapi/v1/img2img", json=payload, timeout=300)
        r.raise_for_status()

        return {"generated_image": r.json()["images"][0]}
    except Exception as e:                       # RunPod сам переведёт в FAILED
        return {"error": str(e)}

# регистрируем воркер
runpod.serverless.start({"handler": handler})    # обязательно! :contentReference[oaicite:0]{index=0}
