#!/bin/bash
# Run vLLM container with Nemotron Nano VL model

# Launch Docker container with NVIDIA GPU support
docker run --runtime nvidia --gpus all \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v ~/.cache/vllm:/root/.cache/vllm \
      -p 8000:8000 \
      --env "HUGGING_FACE_HUB_TOKEN=<YOUR_HUGGINGFACE_TOKEN>" \
      --env "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas" \
      vllm:25.10 \
      vllm serve nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD --load-format fastsafetensors --trust-remote-code --quantization modelopt_fp4 --max-model-len 24000 --gpu-memory-utilization 0.3
