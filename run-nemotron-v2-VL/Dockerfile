# Base image: NVIDIA vLLM container with CUDA support
FROM nvcr.io/nvidia/vllm:25.10-py3

# Set working directory for all subsequent operations
WORKDIR /workspace

# Clone the vLLM repository from GitHub
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm

# Apply custom patch for Nemotron v2 VL support
COPY patch1.patch patch1.patch
RUN patch -p1 patch1.patch

# Configure CUDA architecture for Spark GPUs (12.1, 12.0f, 12.1a)
RUN export TORCH_CUDA_ARCH_LIST=12.1a
# Set path to CUDA PTXAS compiler for Triton
RUN export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# Add CUDA binaries to PATH
RUN export PATH=/usr/local/cuda/bin:$PATH
# Add CUDA libraries to library path
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build and install vLLM:
# 1. Configure to use existing PyTorch installation
# 2. Install build dependencies
# 3. Install vLLM in editable mode without build isolation
RUN python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
    pip install --no-build-isolation -e .

# Expose port 8000 for vLLM API server
EXPOSE 8000

# Default command: start bash shell
CMD ["/bin/bash"]
