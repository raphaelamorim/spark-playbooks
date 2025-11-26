# NVIDIA Forum Scraper & Fine-tuning Pipeline

Tools to scrape NVIDIA Developer Forum questions, enrich them using a local LLM, and fine-tune GPT-OSS-20B on NVIDIA DGX Spark hardware.

## Features

- **Scraper**: Downloads all questions with complete thread data from the NVIDIA Developer Forum.
- **Dataset Creation**: Enriches raw forum threads using a local LLM to create high-quality Q&A pairs.
- **Fine-tuning**: Scripts and Docker configuration to fine-tune GPT-OSS-20B using Unsloth on DGX Spark.
- **Analysis**: Tools to analyze the downloaded forum data.

## Requirements

- Python 3.8+
- Docker (for fine-tuning)
- Access to a local LLM server (e.g., via LM Studio) for dataset enrichment

## Installation

1. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Scrape Forum Questions

Download questions from the NVIDIA Developer Forum (DGX Spark GB10 category).

```bash
# Basic usage (downloads to all_questions/)
python download_nvidia_forum.py

# Custom output directory
python download_nvidia_forum.py -o my_questions

# Adjust rate limiting (delay in seconds)
python download_nvidia_forum.py -d 2.0
```

### 2. Create & Enrich Dataset

Convert downloaded questions into a ShareGPT-style JSON dataset. This step uses a local LLM to clean and summarize the threads.

1.  **Configure LLM Servers:**
    Ensure `llm_config.json` is configured with your local LLM endpoints (e.g., LM Studio).

    ```json
    {
        "servers": [
            {
                "url": "http://localhost:1234/v1/chat/completions",
                "model": "gpt-oss-20b",
                "timeout": 300,
                "max_tokens": 3000,
                "temperature": 0.5
            }
        ]
    }
    ```

2.  **Run Dataset Creation:**

    ```bash
    python create_dataset.py
    ```

    This will process the JSON files in `all_questions/` and save the enriched dataset to `dataset/nvidia_solved_questions_enriched_llm.json`.

### 3. Analyze Data (Optional)

Analyze the downloaded questions using `analyze_questions.py`.

```bash
# Show statistics
python analyze_questions.py -s

# Search questions
python analyze_questions.py -q "GPU"
```

## Fine-tuning on DGX Spark

This section explains how to fine-tune the GPT-OSS-20B model using Unsloth on NVIDIA DGX Spark hardware using the generated dataset.

### 1. Build the Docker Image

Use the provided `Dockerfile.dgx_spark` to build the image:

```bash
docker build -f Dockerfile.dgx_spark -t unsloth-dgx-spark .
```

### 2. Launch the Container

Run the container with GPU access and volume mounts:

```bash
docker run -it \
    --gpus=all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):$(pwd) \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -w $(pwd) \
    unsloth-dgx-spark
```

### 3. Run Fine-tuning

Inside the container, run the fine-tuning script:

```bash
python3 finetune_gpt_oss_spark.py
```

This script will:
1. Load the `unsloth/gpt-oss-20b` model.
2. Load the `dataset/nvidia_solved_questions_enriched_llm.json` dataset.
3. Fine-tune the model using LoRA.
4. Save the fine-tuned adapters to `lora_model/`.

## Export to Ollama

After fine-tuning, you can export the model to GGUF format and run it locally using Ollama.

### 1. Export to GGUF

Inside the Docker container (where you ran the fine-tuning), run the export script:

```bash
python3 export_to_ollama.py
```

This will:
1. Merge the LoRA adapters with the base model.
2. Convert the model to GGUF format (quantized to q4_k_m).
3. Generate a `Modelfile`.
4. Save the output (e.g., `gpt-oss-20b.MXFP4.gguf`) in the current directory.

### 2. Import to LM Studio

You can also import the GGUF model directly into LM Studio:

```bash
lms import gpt-oss-20b.MXFP4.gguf
```

### 3. Create Ollama Model

Once the GGUF file and Modelfile are generated, you can create the Ollama model. You can do this inside the container (if Ollama is installed) or on your host machine.

```bash
./create_ollama_model.sh
```

This script will:
1. Detect the generated GGUF file and Modelfile.
2. Run `ollama create gpt-oss-spark -f Modelfile`.

### 4. Run the Model

You can now chat with your fine-tuned model:

```bash
ollama run gpt-oss-spark
```

## Cleanup

To remove build artifacts, generated models, and temporary files (preserving datasets), run:

```bash
./cleanup.sh
```

## License

This project is provided for educational and research purposes. Please respect NVIDIA's terms of service.
