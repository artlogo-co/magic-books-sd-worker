FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    libcairo2-dev \
    google-perftools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /workspace/stable-diffusion-webui
WORKDIR /workspace/stable-diffusion-webui

RUN python3 -m venv venv
COPY requirements.txt /workspace/requirements.txt

RUN . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install "setuptools==69.5.1" wheel && \
    grep -v "^clip" /workspace/requirements.txt > /tmp/req_no_clip.txt && \
    pip install -r /tmp/req_no_clip.txt && \
    pip install --no-build-isolation \
        "clip @ https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip#sha256=b5842c25da441d6c581b53a5c60e0c2127ebafe0f746f8e15561a006c6c3be6a"

RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet

RUN mkdir -p models/Stable-diffusion
RUN wget -O models/Stable-diffusion/realvisxlV50_v40Bakedvae.safetensors \
    "https://civitai.com/api/download/models/344487?type=Model&format=SafeTensor&size=pruned&fp=fp16"

RUN mkdir -p extensions/sd-webui-controlnet/models
RUN wget -O extensions/sd-webui-controlnet/models/ip-adapter_instant_id_sdxl.bin \
    "https://huggingface.co/OreX/ControlNet/resolve/main/ip-adapter_instant_id_sdxl.bin"
RUN wget -O extensions/sd-webui-controlnet/models/control_instant_id_sdxl.safetensors \
    "https://huggingface.co/OreX/ControlNet/resolve/main/control_instant_id_sdxl.safetensors"
RUN wget -O extensions/sd-webui-controlnet/models/ip-adapter-plus_sdxl_vit-h.safetensors \
    "https://huggingface.co/OreX/ControlNet/resolve/main/ip-adapter-plus_sdxl_vit-h.safetensors"

RUN mkdir -p models/Lora
RUN wget -O models/Lora/lcm-lora-sdxl.safetensors \
    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors"

RUN mkdir -p models/VAE
RUN wget -O models/VAE/sdxl-vae-fp16-fix.safetensors \
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors"

ENV PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
COPY test_input.json .
COPY src/handler.py .
COPY src/start.sh .
RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]