import base64
import os
import subprocess
import time
from typing import Dict, Any

import requests
from requests.adapters import HTTPAdapter, Retry
import runpod

SD_WEBUI_URL = "http://127.0.0.1:7860"

# Create a session with retries
sd_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
sd_session.mount('http://', HTTPAdapter(max_retries=retries))

# ───────────────────────────────── helpers ────────────────────────────────────
def encode(b: bytes) -> str:
    return base64.b64encode(b).decode()

def wait_for_service() -> None:
    """
    Check if the service is ready to receive requests.
    """
    print("Waiting for WebUI API Service to be ready...")
    retries = 0
    while True:
        try:
            response = sd_session.get(f"{SD_WEBUI_URL}/docs", timeout=120)
            if response.status_code == 200:
                print("WebUI API Service is ready!")
                return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:  # Log every 15 retries to avoid spam
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print(f"Error while waiting for service: {err}")
        time.sleep(0.2)

def start_webui() -> None:
    """
    Starts AUTOMATIC1111 in the background once during worker initialization.
    """
    if os.getenv("_WEBUI_STARTED"):
        return

    os.environ["_WEBUI_STARTED"] = "1"
    print("Starting Stable Diffusion WebUI...")
    subprocess.Popen(
        ["bash", "-c", ". venv/bin/activate && python launch.py ${COMMANDLINE_ARGS}"],
        cwd="/workspace/stable-diffusion-webui",
    )
    wait_for_service()

# ───────────────────────────────── handler ───────────────────────────────────
def handler(job: Dict[str, Any]):
    j = job["input"]
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

        r = sd_session.post(f"{SD_WEBUI_URL}/sdapi/v1/img2img", json=payload, timeout=300)
        r.raise_for_status()

        return {"generated_image": r.json()["images"][0]}
    except Exception as e:
        return {"error": str(e)}

# Initialize the service before starting the handler
if __name__ == "__main__":
    start_webui()  # This will also wait for the service to be ready
    print("Starting RunPod worker...")
    runpod.serverless.start({"handler": handler})
