#!/bin/bash
# Run vLLM container with Nemotron Nano VL model

# Launch Docker container with NVIDIA GPU support
docker run --runtime nvidia --gpus all \
      # Increase memory lock and stack size limits for GPU operations
      --ulimit memlock=-1 --ulimit stack=67108864 \
      # Mount HuggingFace cache to avoid re-downloading models
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      # Mount vLLM cache for compiled kernels and other artifacts
      -v ~/.cache/vllm:/root/.cache/vllm \
      # Expose port 8000 for vLLM API server
      -p 8000:8000 \
      # Set HuggingFace token for model access
      --env "HUGGING_FACE_HUB_TOKEN=<YOUR_HUGGINGFACE_TOKEN>" \
      # Set Triton PTXAS compiler path for CUDA compilation
      --env "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas" \
      # Use the vllm:25.10 image built by build.sh
      vllm:25.10 \
      # Serve Nemotron Nano VL 12B model with FP4 quantization
      # --trust-remote-code: Allow execution of custom model code
      # --quantization modelopt_fp4: Use FP4 quantization for reduced memory
      # --max-model-len 24000: Set maximum sequence length
      # --gpu-memory-utilization 0.3: Use 30% of GPU memory for model weights
      vllm serve nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD --trust-remote-code --quantization modelopt_fp4 --max-model-len 24000 --gpu-memory-utilization 0.3