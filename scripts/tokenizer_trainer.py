from google.cloud import storage
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# --- CONFIGURATION ---
OUTPUT_FILE = "tokenizer.json"
BUCKET_NAME = "mhla-from-scratch"

def get_training_corpus():
    
    # 1. English WikiText
    print("Streaming WikiText (En)...")
    ds_en = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    for i, item in enumerate(ds_en):
        if i > 50000: break
        if item['text'].strip(): yield item['text']

    # 2. Thai Wiki
    print("Streaming Thai Wiki...")
    try:
        ds_th = load_dataset("pythainlp/thai-wiki-dataset-v3", split="train", streaming=True)
        for i, item in enumerate(ds_th):
            if i > 30000: break
            text = item.get('text', item.get('detail', ''))
            if text.strip(): yield text
    except Exception as e:
        print(f"Skipping Thai Wiki: {e}")

    # 3. Medical Chat (Thai)
    print("Streaming Medical Chat...")
    ds_med = load_dataset("PHNG/chatmed-thaigpt1k-th", split='train')
    for row in ds_med:
        yield (
            f"<|im_start|>user\n{row.get('instruction','')}\n{row.get('input','')}<|im_end|>\n"
            f"<|im_start|>assistant\n{row.get('output','')}<|im_end|>"
        )

    # 4. LMSYS Chat (En)
    print("Streaming LMSYS Chat...")
    ds_lmsys = load_dataset("ytz20/LMSYS-Chat-GPT-5-Chat-Response", split='train', streaming=True)
    for i, data in enumerate(ds_lmsys):
        if i > 20000: break
        try:
            yield (
                f"<|im_start|>user\n{data['content'][0]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n{data.get('teacher_response', '')}<|im_end|>"
            )
        except: continue

print("Setting up Tokenizer...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ["<|im_start|>", "<|im_end|>", "<|pad|>"]

trainer = trainers.BpeTrainer(
    vocab_size=64000, 
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

print("Starting Training...")
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

print(f"Saving locally to {OUTPUT_FILE}...")
tokenizer.save(OUTPUT_FILE)

def upload_to_bucket(blob_name, file_path, bucket_name):
    
    try:
        print(f"Uploading {file_path} to gs://{bucket_name}/{blob_name}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        print("Upload success")

    except Exception as e:
        print(f"Upload Failed: {e}")
        print("Tip: Run 'gcloud auth application-default login' if permission is denied.")

upload_to_bucket(OUTPUT_FILE, OUTPUT_FILE, BUCKET_NAME)