#!/bin/bash
# Build Docker image for vLLM with Nemotron v2 VL support
# - Tags the image as vllm:25.10
# - max_jobs=66: Sets the maximum number of parallel compilation jobs
# - nvcc_threads=2: Limits NVCC compiler threads to avoid memory issues during build
docker build -t vllm:25.10 . --build-arg max_jobs=66 --build-arg nvcc_threads=2