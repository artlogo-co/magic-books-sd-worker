import base64
import os
import subprocess
import time
from typing import Any, Dict
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

# Create a session with retries
sd_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
sd_session.mount('http://', HTTPAdapter(max_retries=retries))
s3_client_kwargs = {"endpoint_url": AWS_S3_ENDPOINT} if AWS_S3_ENDPOINT else {}
s3_client = boto3.client("s3", **s3_client_kwargs)

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

def try_request_with_retries(payload, max_retries=2, delay_ms=20):
    """
    Try to make the request with specified number of retries and delay between attempts.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(delay_ms / 1000)
                
            response = sd_session.post(f"{SD_WEBUI_URL}/sdapi/v1/img2img", json=payload, timeout=300)
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
        
        response = try_request_with_retries(payload)

        if "images" in response:
            response["images"] = upload_images_to_s3(response["images"])

        return response
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    start_webui()
    print("Starting RunPod worker...")
    runpod.serverless.start({"handler": handler})
