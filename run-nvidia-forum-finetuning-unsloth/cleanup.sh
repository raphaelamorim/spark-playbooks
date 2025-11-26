#!/bin/bash

echo "Cleaning up build artifacts, models, and caches..."

# Fine-tuning artifacts
if [ -d "lora_model" ]; then
    rm -rf lora_model
    echo "Removed lora_model/"
fi

if [ -d "outputs" ]; then
    rm -rf outputs
    echo "Removed outputs/"
fi

if [ -d "unsloth_compiled_cache" ]; then
    rm -rf unsloth_compiled_cache
    echo "Removed unsloth_compiled_cache/"
fi

# Export artifacts
if [ -d "ollama_model" ]; then
    rm -rf ollama_model
    echo "Removed ollama_model/"
fi

if ls *.gguf 1> /dev/null 2>&1; then
    rm *.gguf
    echo "Removed *.gguf files"
fi

if [ -f "Modelfile" ]; then
    rm Modelfile
    echo "Removed Modelfile"
fi

# Dependencies built locally
if [ -d "llama.cpp" ]; then
    rm -rf llama.cpp
    echo "Removed llama.cpp/"
fi

# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Removed __pycache__ directories"

echo "Cleanup Done! (Datasets in 'all_questions/' and 'dataset/' were preserved)"
