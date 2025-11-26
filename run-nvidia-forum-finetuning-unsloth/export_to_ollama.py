from unsloth import FastLanguageModel
import os

def export():
    # 1. Configuration
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # 2. Load Model (Base + Adapter)
    print("Loading model and adapter from 'lora_model'...")
    # Ensure lora_model exists
    if not os.path.exists("lora_model"):
        print("Error: 'lora_model' directory not found. Please run finetune_gpt_oss_spark.py first.")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        "lora_model", # This loads the base model + the adapter saved in lora_model
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Export to GGUF
    # Unsloth handles the conversion and merging.
    # This will create a file inside the 'ollama_model' directory.
    print("Exporting to GGUF for Ollama...")
    model.save_pretrained_gguf("ollama_model", tokenizer, quantization_method = "q4_k_m")
    print("Done! GGUF model saved in 'ollama_model' directory.")

if __name__ == "__main__":
    export()
