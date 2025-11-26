from unsloth import FastLanguageModel
# from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import os

def train():
    # 1. Configuration
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 2. Load Model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gpt-oss-20b", # or "unsloth/gpt-oss-20b-BF16" if load_in_4bit=False
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Load and Format Dataset
    print("Loading dataset...")
    dataset_path = "dataset/nvidia_solved_questions_enriched.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run create_dataset.py first.")
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Standardize ShareGPT format
    # dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            formatted_convo = []
            for msg in convo:
                role = msg["from"]
                content = msg["value"]
                
                # Map 'human'/'gpt' to 'user'/'assistant'
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                    
                # Inject thought using OpenAI Harmony format
                if "thought" in msg and msg["thought"]:
                    # Structure: <|start|> thought <|message|> content <|return|>
                    content = f"<|start|>\n{msg['thought']}\n<|message|>\n{content}\n<|return|>"
                
                formatted_convo.append({"role": role, "content": content})
            
            text = tokenizer.apply_chat_template(formatted_convo, tokenize = False, add_generation_prompt = False)
            texts.append(text)
            
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. Configure LoRA
    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # 5. Train
    print("Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Set to None for full training
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer.train()

    # 6. Save
    print("Saving model...")
    model.save_pretrained("lora_model") # Local saving
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    print("Done!")

if __name__ == "__main__":
    train()
