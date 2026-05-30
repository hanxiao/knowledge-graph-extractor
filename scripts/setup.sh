#!/bin/bash
# One-shot setup for KI Extractor on a GCP L4 GPU instance
# Usage: bash scripts/setup.sh
set -e

echo "=== Installing Docker ==="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes."
fi

echo "=== Installing NVIDIA Container Toolkit ==="
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

echo "=== Downloading model ==="
# Active: LFM2.5-8B-A1B (1B-active MoE), Q4_K_M (~5GB). Fast on-device extractor.
# To use Qwen3.6-35B-A3B Q3 instead (higher quality / verbatim grounding), swap
# the hf_hub_download below to unsloth/Qwen3.6-35B-A3B-MTP-GGUF :
# Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf and update docker-compose.yml + app.py sampling.
mkdir -p models
if [ ! -f models/LFM2.5-8B-A1B-Q4_K_M.gguf ]; then
    pip install -q huggingface-hub
    # use the Python API (the hf/huggingface-cli console script is often not on PATH)
    python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('LiquidAI/LFM2.5-8B-A1B-GGUF', \
'LFM2.5-8B-A1B-Q4_K_M.gguf', local_dir='models')"
else
    echo "Model already downloaded."
fi

echo "=== Starting services ==="
sudo docker compose up -d --build

echo "=== Waiting for llama-server ==="
for i in $(seq 1 60); do
    if curl -s http://localhost:8080/health | grep -q ok; then
        echo "llama-server ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

IP=$(curl -s ifconfig.me)
echo ""
echo "=== Deployment complete ==="
echo "KI Extractor UI:  http://$IP:3000"
echo "llama-server API: http://$IP:8080"
echo ""
echo "To stop: docker compose down"
echo "To stop the GCP instance:"
echo "  gcloud compute instances stop <INSTANCE> --project=<PROJECT> --zone=<ZONE>"
