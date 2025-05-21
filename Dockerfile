# Базовый образ с CUDA и Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

# 1) WebUI + зависимости
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
WORKDIR /workspace/stable-diffusion-webui
RUN python3 -m venv venv
RUN . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install insightface && \
    pip install runpod requests

# Установка ControlNet расширения
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet

# Загрузка модели SDXL
RUN mkdir -p models/Stable-diffusion
RUN wget -O models/Stable-diffusion/realvisxlV50_v40Bakedvae.safetensors "https://civitai.com/api/download/models/344487?type=Model&format=SafeTensor&size=pruned&fp=fp16"

# Загрузка моделей ControlNet
RUN mkdir -p extensions/sd-webui-controlnet/models
RUN wget -O extensions/sd-webui-controlnet/models/ip-adapter_instant_id_sdxl.bin "https://huggingface.co/OreX/ControlNet/resolve/main/ip-adapter_instant_id_sdxl.bin"
RUN wget -O extensions/sd-webui-controlnet/models/control_instant_id_sdxl.safetensors "https://huggingface.co/OreX/ControlNet/resolve/main/control_instant_id_sdxl.safetensors"
RUN wget -O extensions/sd-webui-controlnet/models/ip-adapter-plus_sdxl_vit-h.safetensors "https://huggingface.co/OreX/ControlNet/resolve/main/ip-adapter-plus_sdxl_vit-h.safetensors"

ENV COMMANDLINE_ARGS="--listen --enable-insecure-extension-access --no-half-vae --xformers --api"

# 2) worker
WORKDIR /workspace
COPY src/handler.py .

# Одним процессом: webui в фоне + worker
# CMD ["python", "-u", "handler.py"]
# Запускаем handler.py используя python из venv stable-diffusion-webui
CMD ["/workspace/stable-diffusion-webui/venv/bin/python", "-u", "handler.py"]
