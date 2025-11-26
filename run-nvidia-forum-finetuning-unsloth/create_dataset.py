import json
import glob
import os
import re
import requests
import argparse
import queue
import threading
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Slang and cleanup dictionary
SLANG_MAP = {
    r"\bu\b": "you",
    r"\bplz\b": "please",
    r"\bpls\b": "please",
    r"\bthx\b": "thanks",
    r"\bty\b": "thank you",
    r"\bcuz\b": "because",
    r"\bcos\b": "because",
    r"\br\b": "are",
    r"\bur\b": "your",
    r"\bidk\b": "I do not know",
    r"\bimo\b": "in my opinion",
    r"\bimho\b": "in my humble opinion",
    r"\bfyi\b": "for your information",
    r"\btw\b": "by the way",
}

def load_llm_config():
    config_path = "llm_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration if file doesn't exist
        return {
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

LLM_CONFIG = load_llm_config()

def clean_text(text):
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove emojis (basic range)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Replace slang
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # Remove signature-like lines and excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip common greetings/signatures if they are on their own line
        lower_line = line.lower()
        if lower_line in ['hi', 'hello', 'thanks', 'thank you', 'best regards', 'cheers']:
            continue
        if lower_line.startswith('sent from my'):
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def call_llm_api(payload, server_config):
    try:
        response = requests.post(
            server_config["url"], 
            json=payload, 
            timeout=server_config.get("timeout", 300)
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"LLM request failed on {server_config['url']}: {e}")
        raise e

def parse_llm_response(result):
    content = result['choices'][0]['message']['content']
    
    # Try to parse JSON from the content
    # Sometimes LLMs wrap JSON in ```json ... ```
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        return json.loads(json_str)
    else:
        return json.loads(content)

def enrich_with_llm(thread_data, server_config, all_servers, url):
    """
    Calls the local LLM to summarize and clean the conversation.
    Expects thread_data to be a list of dicts with 'role' and 'content'.
    """
    system_prompt = (
        "You are an expert technical writer and AI assistant. "
        "Your task is to convert a forum discussion into a high-quality, professional training dataset entry. "
        "You must discard irrelevant information, slang, emoticons, and curse words. "
        "Synthesize the information into a coherent Q&A or conversation. "
        "The final answer should be comprehensive, incorporating the accepted solution and any other relevant details from the thread. "
        "Ensure the tone is professional, concise, and the solution is easy to follow. "
        "\n\n"
        "IMPORTANT: You must also capture your thinking process before generating the final response. "
        "Explain your reasoning, how you selected the relevant information, and how you structured the answer. "
        "\n\n"
        "ADDITIONAL TASK: If the thread contains links to other relevant discussions that are confirmed to be working and useful, "
        "you may generate additional separate conversation entries for those topics if you can infer enough context. "
        "\n\n"
        "The output MUST be a valid JSON object with a single key 'conversations_list', which is a list of conversation objects. "
        "Each conversation object must have a 'conversations' key, which is a list of messages. "
        "Each message object must have: "
        "- 'from': either 'human' or 'gpt' "
        "- 'value': the text content of the message "
        "- 'thought': (ONLY for 'gpt' role) your thinking process and reasoning steps for this response. "
        "\n\n"
        "Ensure the conversation starts with 'human'. "
        "Do not include any markdown formatting or explanations outside the JSON. "
        "Do not output internal thoughts or reasoning traces outside the JSON structure. Provide only the final JSON."
    )
    
    user_prompt = f"Here is the raw forum discussion from {url}:\n{json.dumps(thread_data, indent=2)}\n\nPlease convert this into the specified JSON format."

    payload = {
        "model": server_config.get("model", "gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": server_config.get("temperature", 0.5),
        "max_tokens": server_config.get("max_tokens", 8192), # Increased limit for thoughts
        "stream": False
    }

    # Try primary server with retries
    for attempt in range(3):
        try:
            result = call_llm_api(payload, server_config)
            return parse_llm_response(result)
        except Exception:
            if attempt < 2:
                time.sleep(2) # Wait before retry
            continue

    print(f"Primary server {server_config['url']} failed 3 times. Attempting failover...")

    # Failover to other servers
    for other_server in all_servers:
        if other_server['url'] == server_config['url']:
            continue
            
        print(f"Trying failover server: {other_server['url']}")
        try:
            # Update payload model if needed, but usually we keep the same request
            # Assuming compatible APIs
            payload["model"] = other_server.get("model", payload["model"])
            
            result = call_llm_api(payload, other_server)
            return parse_llm_response(result)
        except Exception:
            continue
            
    print("All servers failed to process the request.")
    return None

def process_file(file_path, server_config, all_servers, tmp_dir):
    # Check if tmp file exists to avoid re-processing
    base_name = os.path.basename(file_path)
    tmp_file_path = os.path.join(tmp_dir, base_name)
    
    if os.path.exists(tmp_file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check for accepted answer
    has_accepted = data.get('topic_accepted_answer') is True or \
                   (isinstance(data.get('accepted_answer'), dict))
    
    if not has_accepted:
        return []

    posts = data.get('post_stream', {}).get('posts', [])
    if not posts:
        return []

    # Identify OP and Answerer
    op_id = None
    
    # Find OP (first post)
    for post in posts:
        if post.get('post_number') == 1:
            op_id = post.get('user_id')
            break
            
    if op_id is None:
        return []

    # Find Answerer
    answer_post = None
    for post in posts:
        if post.get('accepted_answer') is True:
            answer_post = post
            break
            
    if not answer_post and isinstance(data.get('accepted_answer'), dict):
        accepted_post_num = data['accepted_answer'].get('post_number')
        for post in posts:
            if post.get('post_number') == accepted_post_num:
                answer_post = post
                break
                
    if not answer_post:
        return []
        
    answerer_id = answer_post.get('user_id')
    
    # Collect relevant posts (OP and Answerer only to keep it clean)
    relevant_posts = []
    for post in posts:
        uid = post.get('user_id')
        # We include OP and Answerer, and maybe others if they are relevant?
        # For now, let's stick to OP and Answerer to avoid noise, or maybe include all and let LLM filter?
        # The prompt says "discard irrelevant information", so giving more context to LLM is better.
        # But to save tokens, maybe just OP and Answerer is safer?
        # User said "all comments and accepted answers", so I should include all posts.
        relevant_posts.append(post)
            
    # Sort by post_number
    relevant_posts.sort(key=lambda x: x.get('post_number'))
    
    # Prepare data for LLM
    thread_data = []
    title = data.get('title', '')
    thread_data.append({"role": "system", "content": f"Thread Title: {title}"})
    
    total_chars = 0
    MAX_CHARS = 60000 # Increased to ~15k tokens based on 60k context window
    
    accepted_post_number = answer_post.get('post_number')

    for post in relevant_posts:
        if total_chars > MAX_CHARS:
            break
            
        role = "human" if post.get('user_id') == op_id else "gpt"
        
        user_label = "User"
        if post.get('user_id') == op_id:
            user_label = "Original Poster"
        
        if post.get('post_number') == accepted_post_number:
            user_label = "Solution Provider (ACCEPTED ANSWER)"
        elif post.get('user_id') == answerer_id:
            user_label = "Solution Provider"
        elif post.get('user_id') != op_id:
            user_label = "Other User"
            
        text = clean_text(post.get('cooked', ''))
        if text:
            # Truncate very long posts
            if len(text) > 5000:
                text = text[:5000] + "... (truncated)"
            
            thread_data.append({
                "user": user_label,
                "text": text
            })
            total_chars += len(text)

    # Call LLM
    url = f"https://forums.developer.nvidia.com/t/{data.get('slug')}/{data.get('id')}"
    enriched_data = enrich_with_llm(thread_data, server_config, all_servers, url)
    
    entries = []
    
    if enriched_data:
        # Handle new list format
        if 'conversations_list' in enriched_data:
            for conv_obj in enriched_data['conversations_list']:
                if 'conversations' in conv_obj:
                    entries.append({
                        "conversations": conv_obj['conversations'],
                        "reference": url,
                        "type": "llm_enriched"
                    })
        # Handle legacy single conversation format (fallback)
        elif 'conversations' in enriched_data:
            entries.append({
                "conversations": enriched_data['conversations'],
                "reference": url,
                "type": "llm_enriched"
            })
    
    if not entries:
        # Fallback to heuristic method if LLM fails
        # ...existing code for heuristic method...
        # (I will re-implement the heuristic method here as fallback)
        
        # Build conversation (Heuristic)
        conversation = []
        current_role = None
        current_text = []
        
        # Filter for heuristic: only OP and Answerer
        heuristic_posts = [p for p in relevant_posts if p.get('user_id') in [op_id, answerer_id]]
        
        for i, post in enumerate(heuristic_posts):
            role = "human" if post.get('user_id') == op_id else "gpt"
            text = clean_text(post.get('cooked', ''))
            
            if not text:
                continue
                
            if post.get('post_number') == 1:
                text = f"{title}\n\n{text}"
                
            if role == current_role:
                current_text.append(text)
            else:
                if current_role:
                    conversation.append({
                        "from": current_role,
                        "value": "\n\n".join(current_text)
                    })
                current_role = role
                current_text = [text]
                
        if current_role and current_text:
            conversation.append({
                "from": current_role,
                "value": "\n\n".join(current_text)
            })
            
        if conversation and conversation[0]['from'] == 'human':
             entries.append({
                "conversations": conversation,
                "reference": f"https://forums.developer.nvidia.com/t/{data.get('slug')}/{data.get('id')}",
                "type": "heuristic_fallback"
            })

    # Save to tmp file
    if entries:
        with open(tmp_file_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)

    return entries

def create_dataset(limit=None):
    questions_dir = "all_questions"
    output_dir = "dataset"
    tmp_dir = os.path.join(output_dir, "tmp")
    output_file = os.path.join(output_dir, "nvidia_solved_questions_enriched_llm.json")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    json_files = glob.glob(os.path.join(questions_dir, "*.json"))
    
    if limit:
        json_files = json_files[:limit]
    
    servers = LLM_CONFIG.get("servers", [])
    if not servers:
        print("No LLM servers configured in llm_config.json")
        return

    print(f"Processing {len(json_files)} files with LLM enrichment using {len(servers)} servers...")
    
    file_queue = queue.Queue()
    for file_path in json_files:
        # Check if already processed in tmp
        base_name = os.path.basename(file_path)
        tmp_file_path = os.path.join(tmp_dir, base_name)
        if not os.path.exists(tmp_file_path):
            file_queue.put(file_path)
        else:
            print(f"Skipping {base_name} (already processed)")
        
    lock = threading.Lock()
    processed_count = 0
    total_files = len(json_files)
    
    def worker(server_config):
        nonlocal processed_count
        while True:
            try:
                file_path = file_queue.get_nowait()
            except queue.Empty:
                break
            
            try:
                # Pass all servers for failover
                process_file(file_path, server_config, servers, tmp_dir)
                
                with lock:
                    processed_count += 1
                    print(f"[{processed_count}/{total_files}] Finished processing {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            finally:
                file_queue.task_done()
    
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = [executor.submit(worker, server) for server in servers]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed: {e}")
            
    # Assemble final dataset from tmp files
    print("Assembling final dataset from temporary files...")
    dataset = []
    tmp_files = glob.glob(os.path.join(tmp_dir, "*.json"))
    for tmp_file in tmp_files:
        try:
            with open(tmp_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)
                dataset.extend(entries)
        except Exception as e:
            print(f"Error reading tmp file {tmp_file}: {e}")

    print(f"Generated {len(dataset)} conversation entries.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset from forum questions.")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process.")
    args = parser.parse_args()
    
    create_dataset(limit=args.limit)
