import base64
import os
import subprocess
import time
from typing import Any, Dict, Optional
from uuid import uuid4

import boto3
import requests
from requests.adapters import HTTPAdapter, Retry
import runpod

SD_WEBUI_URL = "http://127.0.0.1:7860"
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_S3_PREFIX = os.getenv("AWS_S3_PREFIX", "")
AWS_S3_URL_EXPIRY = int(os.getenv("AWS_S3_URL_EXPIRY", "3600"))
AWS_S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")

# LCM acceleration toggle via env var (set LCM_ENABLED=1 to activate)
LCM_ENABLED = os.getenv("LCM_ENABLED", "0") == "1"
LCM_LORA_NAME = os.getenv("LCM_LORA_NAME", "lcm-lora-sdxl")
LCM_LORA_WEIGHT = float(os.getenv("LCM_LORA_WEIGHT", "0.9"))
LCM_STEPS = int(os.getenv("LCM_STEPS", "10"))
LCM_CFG_SCALE = float(os.getenv("LCM_CFG_SCALE", "1.8"))

sd_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
sd_session.mount('http://', HTTPAdapter(max_retries=retries))
s3_client_kwargs = {"endpoint_url": AWS_S3_ENDPOINT} if AWS_S3_ENDPOINT else {}
s3_client = boto3.client("s3", **s3_client_kwargs)

# ───────────────────────────────── helpers ────────────────────────────────────
def encode(b: bytes) -> str:
    return base64.b64encode(b).decode()


def apply_lcm_acceleration(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Patch payload to use LCM LoRA for faster inference.
    Original values are preserved in '_original' keys for logging.
    """
    patched = payload.copy()

    # --- sampler / scheduler ---
    patched["_original_sampler"] = payload.get("sampler_name")
    patched["_original_scheduler"] = payload.get("scheduler")
    patched["_original_steps"] = payload.get("steps")
    patched["_original_cfg_scale"] = payload.get("cfg_scale")

    patched["sampler_name"] = "LCM"
    patched["scheduler"] = "Karras"
    patched["steps"] = LCM_STEPS
    patched["cfg_scale"] = LCM_CFG_SCALE

    # --- inject LCM LoRA into prompt ---
    lora_tag = f"<lora:{LCM_LORA_NAME}:{LCM_LORA_WEIGHT}>"
    original_prompt = payload.get("prompt", "")

    if lora_tag not in original_prompt:
        patched["prompt"] = f"{original_prompt}, {lora_tag}"

    print(
        f"[LCM] Patched payload: "
        f"steps {patched['_original_steps']} → {LCM_STEPS}, "
        f"cfg {patched['_original_cfg_scale']} → {LCM_CFG_SCALE}, "
        f"sampler {patched['_original_sampler']} → LCM"
    )

    return patched


def strip_internal_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove _original_* debug keys before sending to WebUI."""
    return {k: v for k, v in payload.items() if not k.startswith("_original")}


def _extract_base64_image(value: Any) -> Optional[str]:
    """
    Normalize various "image" representations to the base64 string SD WebUI expects.

    Forge's ControlNet implementation is stricter than A1111's extension: it expects
    unit["image"] to be a base64 string (or null), not a nested object.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    # A1111-style: {"image": "<b64>", "mask": ...}
    if isinstance(value, dict):
        nested = value.get("image")
        if isinstance(nested, str) or nested is None:
            return nested
        return None

    return None


def normalize_controlnet_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix common ControlNet payload incompatibilities between A1111 and Forge.

    - If a ControlNet unit has "image" as a dict (e.g. {"mask": null} or
      {"image": "<b64>", "mask": null}), convert it to Forge-friendly fields:
        unit["image"] = "<b64>" or None
        unit["mask"]  = <mask> (if present)
    """
    alwayson = payload.get("alwayson_scripts")
    if not isinstance(alwayson, dict):
        return payload

    patched = payload.copy()
    patched_alwayson = dict(alwayson)
    changed = False

    for script_name, script_cfg in list(patched_alwayson.items()):
        if not isinstance(script_name, str) or "controlnet" not in script_name.lower():
            continue
        if not isinstance(script_cfg, dict):
            continue

        args = script_cfg.get("args")
        if not isinstance(args, list):
            continue

        new_args = []
        args_changed = False

        for unit in args:
            if not isinstance(unit, dict):
                new_args.append(unit)
                continue

            new_unit = unit.copy()

            if "image" in new_unit and isinstance(new_unit.get("image"), dict):
                image_dict = new_unit.get("image") or {}
                new_unit["image"] = _extract_base64_image(image_dict)

                if "mask" in image_dict and "mask" not in new_unit:
                    new_unit["mask"] = image_dict.get("mask")

                args_changed = True

            new_args.append(new_unit)

        if args_changed:
            new_script_cfg = script_cfg.copy()
            new_script_cfg["args"] = new_args
            patched_alwayson[script_name] = new_script_cfg
            changed = True

    if not changed:
        return payload

    patched["alwayson_scripts"] = patched_alwayson
    return patched


def wait_for_service() -> None:
    print("Waiting for WebUI API Service to be ready...")
    retries = 0
    while True:
        try:
            response = sd_session.get(f"{SD_WEBUI_URL}/sdapi/v1/sd-models", timeout=120)
            if response.status_code == 200:
                try:
                    models = response.json()
                    if isinstance(models, list):
                        print("WebUI API Service is ready!")
                        return
                except ValueError:
                    pass
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                print(f"Service not ready yet. Retry #{retries}...")
        except Exception as err:
            print(f"Error while waiting for service: {err}")
        time.sleep(1 if retries > 30 else 0.2)


def start_webui() -> None:
    if os.getenv("_WEBUI_STARTED"):
        return

    os.environ["_WEBUI_STARTED"] = "1"
    print("Starting Stable Diffusion WebUI...")

    cmd_args = os.getenv(
        "COMMANDLINE_ARGS",
        "--listen --enable-insecure-extension-access --opt-sdp-attention --opt-channelslast --api"
    )

    python_path = "/workspace/stable-diffusion-webui/venv/bin/python"
    launch_script = "/workspace/stable-diffusion-webui/launch.py"

    subprocess.Popen(
        [python_path, launch_script] + cmd_args.split(),
        cwd="/workspace/stable-diffusion-webui",
        env=dict(os.environ, PYTHONUNBUFFERED="1")
    )
    wait_for_service()


def try_request_with_retries(payload, max_retries=2, delay_ms=20):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(delay_ms / 1000)

            response = sd_session.post(
                f"{SD_WEBUI_URL}/sdapi/v1/img2img",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {str(e)}")

    raise last_error


def upload_images_to_s3(images):
    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET env var is required to upload images")

    prefix = AWS_S3_PREFIX.strip("/")
    urls = []

    for image_b64 in images:
        payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        data = base64.b64decode(payload)

        object_key = f"{uuid4()}.png"
        if prefix:
            object_key = f"{prefix}/{object_key}"

        s3_client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=object_key,
            Body=data,
            ContentType="image/png",
        )

        urls.append(
            s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": AWS_S3_BUCKET, "Key": object_key},
                ExpiresIn=AWS_S3_URL_EXPIRY,
            )
        )

    return urls


# ───────────────────────────────── handler ───────────────────────────────────
def handler(job: Dict[str, Any]):
    try:
        wait_for_service()
        payload = job["input"]

        payload = normalize_controlnet_payload(payload)

        # --- LCM acceleration patch ---
        if LCM_ENABLED:
            payload = apply_lcm_acceleration(payload)

        clean_payload = strip_internal_keys(payload)
        response = try_request_with_retries(clean_payload)

        if "images" in response:
            response["images"] = upload_images_to_s3(response["images"])

        # Attach LCM debug info to response if patched
        if LCM_ENABLED:
            response["_lcm_debug"] = {
                "lcm_enabled": True,
                "steps_used": LCM_STEPS,
                "cfg_used": LCM_CFG_SCALE,
                "original_steps": payload.get("_original_steps"),
                "original_cfg": payload.get("_original_cfg_scale"),
                "original_sampler": payload.get("_original_sampler"),
            }

        return response

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    start_webui()
    print("Starting RunPod worker...")
    runpod.serverless.start({"handler": handler})