# Nemotron v2 VL on vLLM

Run the Nemotron Nano VL 12B V2 vision-language model using vLLM with FP4 quantization on NVIDIA DGX Spark systems.

## Overview

This playbook provides a complete setup for deploying NVIDIA's Nemotron Nano VL 12B V2 model using vLLM inference engine. The model is optimized with FP4 quantization for efficient inference on DGX Spark GPUs while maintaining high quality outputs.

### Key Features

- **Vision-Language Model**: Process both text and images
- **FP4 Quantization**: Reduced memory footprint (4-bit precision)
- **OpenAI-Compatible API**: Easy integration with existing applications
- **Optimized for Spark GPUs**: Custom CUDA architecture configuration
- **Production Ready**: Includes health checks, monitoring, and error handling

## Prerequisites

- NVIDIA DGX Spark system with GPU support
- Docker installed with NVIDIA Container Toolkit
- NVIDIA GPU drivers installed
- HuggingFace account with access token (for downloading models)
- Sufficient disk space (~20GB for the Docker image and model cache)

### Verify Prerequisites

```bash
# Check GPU availability
nvidia-smi

# Check Docker with NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check disk space
df -h
```

## Directory Structure

```
run-nemotron-v2-VL/
├── README.md         # This file
├── build.sh          # Script to build the Docker image
├── Dockerfile        # Docker configuration with vLLM and patches
├── patch1.patch      # Custom patch for Nemotron v2 VL support
└── run.sh            # Script to run the vLLM server
```

## Building the Docker Image

The `build.sh` script builds a Docker image with vLLM compiled for NVIDIA DGX Spark GPUs.

### Build Configuration

The build process uses these parameters:
- **max_jobs=66**: Maximum parallel compilation jobs (adjust based on your CPU cores)
- **nvcc_threads=2**: NVCC compiler threads (limited to avoid memory issues during compilation)

### Build Steps

1. Navigate to this directory:
   ```bash
   cd run-nemotron-v2-VL
   ```

2. Make the build script executable (if needed):
   ```bash
   chmod +x build.sh
   ```

3. Run the build script:
   ```bash
   ./build.sh
   ```

4. Wait for the build to complete. This may take 30-60 minutes depending on your system.

### What the Build Does

The Docker build process:
1. Starts from the NVIDIA vLLM 25.10 base image
2. Clones the vLLM repository
3. Applies the custom patch for Nemotron v2 VL support
4. Configures CUDA architecture for Spark GPUs (12.1a)
5. Builds and installs vLLM with the existing PyTorch installation

### Troubleshooting Build Issues

- **Out of memory during build**: Reduce `max_jobs` parameter in `build.sh`
- **CUDA compilation errors**: Ensure NVIDIA drivers are up to date
- **Disk space issues**: Free up space or change Docker's storage location
- **Network timeouts**: Check internet connectivity, the git clone may need retry

## Running the vLLM Server

The `run.sh` script launches a Docker container running the vLLM server with the Nemotron Nano VL 12B V2 model.

### Configuration Steps

Before running, you need to update the HuggingFace token in `run.sh`:

1. Get your HuggingFace token:
   - Visit https://huggingface.co/settings/tokens
   - Create a new token or copy an existing one
   - Ensure you have accepted the model's license agreement at https://huggingface.co/nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD

2. Edit `run.sh` and replace the token:
   ```bash
   --env "HUGGING_FACE_HUB_TOKEN=your_token_here" \
   ```

### Running the Server

1. Make the run script executable (if needed):
   ```bash
   chmod +x run.sh
   ```

2. Launch the vLLM server:
   ```bash
   ./run.sh
   ```

3. The server will start on port 8000 and begin loading the model. Initial startup may take several minutes as the model is downloaded and loaded into GPU memory.

### Server Configuration

The server is configured with these parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD | The vision-language model |
| Quantization | FP4 (modelopt_fp4) | 4-bit floating point quantization |
| Max sequence length | 24,000 tokens | Maximum context window |
| GPU memory utilization | 30% | Percentage of GPU memory for model weights |
| Port | 8000 | HTTP API port |

### Container Features

The Docker container is configured with:
- **GPU Access**: Full access to all GPUs (`--gpus all`)
- **Memory Limits**: Increased ulimits for GPU operations
- **Persistent Caches**:
  - `~/.cache/huggingface`: Model weights and tokenizers (persists across restarts)
  - `~/.cache/vllm`: Compiled CUDA kernels (improves startup time)
- **Environment Variables**: CUDA compilation paths for Triton

## Using the vLLM Server

Once the server is running, you can interact with it via the OpenAI-compatible API.

### Health Check

Check if the server is ready:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### Example API Request

Generate text with the model:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD",
    "prompt": "Describe this image:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Using with Python

Install the OpenAI Python client:
```bash
pip install openai
```

Example code:
```python
from openai import OpenAI

# Create client pointing to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require authentication by default
)

# Generate completion
response = client.completions.create(
    model="nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD",
    prompt="Hello, how are you?",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Chat Completion Example

```python
# Chat-style interaction
response = client.chat.completions.create(
    model="nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you tell me about AI?"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Performance Tuning

### GPU Memory Utilization

Adjust the `--gpu-memory-utilization` parameter in `run.sh`:
- **0.3** (30%): Conservative, leaves memory for other workloads
- **0.5** (50%): Balanced approach
- **0.9** (90%): Maximum performance, uses most available GPU memory

Example:
```bash
# In run.sh, change:
--gpu-memory-utilization 0.5
```

### Sequence Length

Modify `--max-model-len` based on your use case:
- **4,096**: Short conversations, faster inference
- **8,192**: Medium contexts
- **24,000**: Default, good balance
- **32,768**: Maximum context (if supported), higher memory usage

### Batch Size and Throughput

For higher throughput with multiple concurrent requests, add to the `vllm serve` command in `run.sh`:
```bash
--max-num-seqs 256 \
--max-num-batched-tokens 8192
```

### Temperature and Sampling

Control output randomness in your API requests:
- **temperature=0.0**: Deterministic, always picks most likely token
- **temperature=0.7**: Balanced creativity (default)
- **temperature=1.0**: More random and creative
- **top_p=0.9**: Nucleus sampling for quality control

## Monitoring

### View Container Logs

Follow the logs in real-time:
```bash
docker logs -f $(docker ps -q --filter ancestor=vllm:25.10)
```

View last 100 lines:
```bash
docker logs --tail 100 $(docker ps -q --filter ancestor=vllm:25.10)
```

### Check GPU Usage

Monitor GPU utilization:
```bash
nvidia-smi
```

Continuous monitoring (updates every 1 second):
```bash
watch -n 1 nvidia-smi
```

### Monitor Container Resources

```bash
docker stats $(docker ps -q --filter ancestor=vllm:25.10)
```

### Server Metrics

vLLM exposes metrics at:
```bash
curl http://localhost:8000/metrics
```

## Stopping the Server

1. Find the container ID:
   ```bash
   docker ps
   ```

2. Stop the container gracefully:
   ```bash
   docker stop <container_id>
   ```

Or stop all vLLM containers:
```bash
docker stop $(docker ps -q --filter ancestor=vllm:25.10)
```

## Cleanup

### Remove Stopped Containers

```bash
docker rm $(docker ps -aq --filter ancestor=vllm:25.10)
```

### Remove the Docker Image

```bash
docker rmi vllm:25.10
```

### Clear Caches

Free up disk space by clearing downloaded models and compiled kernels:
```bash
# Remove HuggingFace cache
rm -rf ~/.cache/huggingface

# Remove vLLM cache
rm -rf ~/.cache/vllm
```

**Warning**: This will require re-downloading models on next run.

## Troubleshooting

### Server Won't Start

**Issue**: Container exits immediately or fails to start

**Solutions**:
- Check GPU availability: `nvidia-smi`
- Verify Docker NVIDIA runtime: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
- Check port 8000 availability: `lsof -i :8000` or `netstat -tulpn | grep 8000`
- Review container logs: `docker logs <container_id>`

### Out of Memory Errors

**Issue**: `CUDA out of memory` or similar errors

**Solutions**:
1. Reduce `--gpu-memory-utilization` to 0.2 or 0.3
2. Reduce `--max-model-len` to 8192 or 4096
3. Close other GPU applications: `nvidia-smi` to see what's running
4. Ensure no other containers are using the GPU

### Slow Inference

**Issue**: Response times are slower than expected

**Solutions**:
- Increase `--gpu-memory-utilization` if you have available GPU memory
- Ensure the model is fully loaded (check logs for "Loaded model" message)
- Verify GPU isn't being throttled: `nvidia-smi` (check temperature and power)
- Reduce `--max-num-seqs` if you're not serving many concurrent requests
- Check if other processes are competing for GPU resources

### Model Download Issues

**Issue**: Model fails to download or authentication errors

**Solutions**:
- Verify your HuggingFace token is correct and valid
- Ensure you've accepted the model's license agreement
- Check internet connectivity: `ping huggingface.co`
- Verify disk space is available: `df -h`
- Check HuggingFace status: https://status.huggingface.co/

### Build Failures

**Issue**: Docker build fails with compilation errors

**Solutions**:
- Reduce `max_jobs` in `build.sh` (try 32 or 16)
- Ensure sufficient RAM is available (build requires ~64GB+)
- Check NVIDIA driver version: `nvidia-smi`
- Verify CUDA toolkit is accessible in the container
- Review full build logs for specific error messages

## Advanced Configuration

### Custom CUDA Architecture

If you need to target different GPU architectures, modify the `TORCH_CUDA_ARCH_LIST` in the Dockerfile:
```dockerfile
RUN export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
```

### Environment Variables

Additional environment variables you can set in `run.sh`:
```bash
--env "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
--env "VLLM_LOGGING_LEVEL=INFO" \
--env "CUDA_VISIBLE_DEVICES=0,1" \  # Limit to specific GPUs
```

### Multi-GPU Setup

To use multiple GPUs, vLLM automatically detects and uses all available GPUs. You can limit which GPUs to use:
```bash
--env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Nemotron Model Card](https://huggingface.co/nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx/)

## Support

For issues specific to:
- **vLLM**: Check the [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- **Nemotron Model**: Refer to the [HuggingFace model discussions](https://huggingface.co/nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD/discussions)
- **DGX Spark Hardware**: Contact NVIDIA support
- **This Playbook**: Open an issue in the spark-playbooks repository

## Version Information

- **vLLM Version**: 25.10
- **Model**: Nemotron-Nano-VL-12B-V2-FP4-QAD
- **CUDA Architecture**: 12.1a (optimized for Spark)
- **Last Updated**: November 2025
