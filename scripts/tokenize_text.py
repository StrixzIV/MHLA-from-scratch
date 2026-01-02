import torch
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from google.cloud import storage
import os
from tqdm import tqdm

# --- CONFIGURATION ---
BUCKET_NAME = "mhla-from-scratch"
TOKENIZER_FILE = "tokenizer.json"
OUTPUT_DIR = "./processed_data"
BLOCK_SIZE = 2048 

LIMIT_FINEWEB = 100_000   # high-quality English
LIMIT_C4_TH = 100_000     # diverse Thai
LIMIT_CODE = 20_000       # Python code samples
LIMIT_CHAT = 50_000       # Chat logs

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def download_tokenizer_if_needed():
    
    if not os.path.exists(TOKENIZER_FILE):
        
        print(f"📉 Tokenizer not found locally. Downloading from gs://{BUCKET_NAME}/{TOKENIZER_FILE}...")
        
        try:
            blob = bucket.blob(TOKENIZER_FILE)
            blob.download_to_filename(TOKENIZER_FILE)
            print("✅ Tokenizer downloaded.")
        
        except Exception as e:
            print(f"❌ Error downloading tokenizer: {e}")
            print("Make sure you ran the tokenizer trainer and uploaded it first!")
            exit(1)
    
    else:
        print("✅ Tokenizer found locally.")

def upload_directory(local_path, bucket_path):
    
    print(f"☁️ Uploading processed data to gs://{BUCKET_NAME}/{bucket_path}...")
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(bucket_path, relative_path)
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
    print("✅ Upload Complete!")

def format_chat(row):
    """
    Standardizes different datasets into the <|im_start|> format.
    """
    # 1. Chat/Instruct Format (Medical & LMSYS)
    if 'instruction' in row: # Medical
        instruction = row.get('instruction', '')
        inp = row.get('input', '')
        resp = row.get('output', '')
        user_text = f"{instruction}\n\n{inp}" if inp else instruction
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{resp}<|im_end|>"

    if 'content' in row and isinstance(row['content'], list): # LMSYS
        try:
            user = row['content'][0]['content']
            assist = row.get('teacher_response', row['content'][1]['content'])
            return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assist}<|im_end|>"
        except: return ""

    text = row.get('text', row.get('content', ''))
    if len(text) > 20: 
        return f"{text}<|im_end|>"
    
    return ""

def data_generator():
    
    print(f"Streaming FineWeb-Edu (Limit: {LIMIT_FINEWEB})...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= LIMIT_FINEWEB: break
        yield format_chat(row)

    # --- 2. General Thai (C4 Thai + Wiki) ---
    print(f"Streaming C4 Thai (Limit: {LIMIT_C4_TH})...")
    try:
        # C4 is huge, we stream the 'th' subset
        ds = load_dataset("allenai/c4", "th", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= LIMIT_C4_TH: break
            yield format_chat(row)
    except Exception as e:
        print(f"⚠️ Could not load C4 Thai: {e}")

    # --- 3. Code (The Stack Smol) ---
    # Helps with logic and formatting
    print(f"Streaming Python Code (Limit: {LIMIT_CODE})...")
    try:

        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)

        for i, row in enumerate(ds):
            if i >= LIMIT_CODE: break
            yield format_chat(row)

    except Exception as e:
        print(f"⚠️ Could not load Code: {e}")

    print("Streaming Medical Chat...")
    ds = load_dataset("PHNG/chatmed-thaigpt1k-th", split="train")
    for row in ds: yield format_chat(row)

    print("Streaming LMSYS Chat...")
    ds = load_dataset("ytz20/LMSYS-Chat-GPT-5-Chat-Response", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= LIMIT_CHAT: break
        yield format_chat(row)

def tokenize_and_chunk(tokenizer):
    
    current_chunk = []
    dataset_dict = {"input_ids": []}
    
    EOS_TOKEN_ID = tokenizer.token_to_id("<|im_end|>")
    PAD_TOKEN_ID = tokenizer.token_to_id("<|pad|>")

    print("Tokenizing and Packing...")
    # tqdm helps you see progress since streaming has no total length
    for text in tqdm(data_generator()):
        if not text: continue
        
        # Check if tokenizer handles Thai correctly (sometimes adds empty tokens)
        ids = tokenizer.encode(text).ids
        current_chunk.extend(ids)

        while len(current_chunk) >= BLOCK_SIZE:
            block = current_chunk[:BLOCK_SIZE]
            dataset_dict["input_ids"].append(block)
            current_chunk = current_chunk[BLOCK_SIZE:]

    if len(current_chunk) > 0:
        padding = [PAD_TOKEN_ID] * (BLOCK_SIZE - len(current_chunk))
        dataset_dict["input_ids"].append(current_chunk + padding)

    return Dataset.from_dict(dataset_dict)

if __name__ == "__main__":
    
    download_tokenizer_if_needed()
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    final_dataset = tokenize_and_chunk(tokenizer)

    print("Splitting Train/Val...")
    split_dataset = final_dataset.train_test_split(test_size=0.1)
    
    print(f"Saving to {OUTPUT_DIR}...")
    split_dataset.save_to_disk(OUTPUT_DIR)
    
    upload_directory(OUTPUT_DIR, "processed_data")