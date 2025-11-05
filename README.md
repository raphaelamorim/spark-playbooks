# spark-playbooks

A collection of playbooks and recipes for deploying AI models and workloads on NVIDIA DGX Spark systems.

## Overview

This repository contains ready-to-use playbooks for running various AI models and frameworks optimized for NVIDIA DGX Spark GPUs. Each playbook includes Docker configurations, build scripts, and comprehensive documentation.

## Available Playbooks

### 🚀 [Nemotron v2 VL on vLLM](./run-nemotron-v2-VL/)

Run the Nemotron Nano VL 12B V2 vision-language model using vLLM with FP4 quantization.

- **Model**: nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD
- **Framework**: vLLM (optimized for inference)
- **Quantization**: FP4 for reduced memory usage
- **API**: OpenAI-compatible REST API
- **Use Cases**: Vision-language tasks, multimodal AI applications

[📖 View Documentation](./run-nemotron-v2-VL/README.md)

## Repository Structure

```
spark-playbooks/
├── README.md                           # This file
├── run-nemotron-v2-VL/                 # Nemotron VL playbook
│   ├── README.md                       # Playbook-specific documentation
│   ├── build.sh                        # Build script
│   ├── run.sh                          # Run script
│   ├── Dockerfile                      # Docker configuration
│   └── patch1.patch                    # Custom patches
└── [other-playbooks]/                  # Additional playbooks
    └── README.md                       # Playbook-specific documentation
```

## Quick Start

1. **Choose a playbook** from the list above
2. **Navigate to the playbook directory**:
   ```bash
   cd [playbook-name]
   ```
3. **Follow the playbook's README.md** for detailed instructions

## General Prerequisites

All playbooks in this repository require:

- NVIDIA DGX Spark system with GPU support
- Docker installed with NVIDIA Container Toolkit
- NVIDIA GPU drivers (recommended: latest stable version)
- Sufficient disk space (varies by playbook)

### Verify Your Setup

Check that your system is ready:

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check available disk space
df -h
```

## Contributing

We welcome contributions! To add a new playbook:

1. Create a new directory with a descriptive name
2. Include all necessary files (Dockerfile, scripts, patches)
3. Add a comprehensive README.md in the playbook directory
4. Update this main README.md to list your playbook

### Playbook README Template

Each playbook should include:

- **Overview**: What the playbook does
- **Prerequisites**: Specific requirements
- **Build Instructions**: Step-by-step build process
- **Run Instructions**: How to launch and configure
- **Usage Examples**: API calls, code samples
- **Configuration**: Tuning parameters
- **Troubleshooting**: Common issues and solutions

## Additional Resources

- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Documentation](https://docs.docker.com/)

## Support

For playbook-specific issues, refer to the individual playbook's README.md documentation.

For general DGX Spark system issues, contact NVIDIA support.

## License

See [LICENSE](./LICENSE) file for details.
