import os
import time
import torch
from datasets import load_dataset

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from google.cloud import storage
from datasets import load_from_disk

from chat import m, model
from model import GPTLanguageModel, batch_size, learning_rate

BUCKET_NAME = "mhla-from-scratch"
DATA_DIR = "./processed_data"

MAX_ITERS = 5000
EVAL_INTERVAL = 500
SAVE_INTERVAL = 500

def download_data_if_missing():
    
    if not os.path.exists(DATA_DIR):
        
        print(f"📉 Data not found. Downloading from gs://{BUCKET_NAME}/processed_data...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="processed_data")
        
        for blob in blobs:
            local_path = blob.name
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            
        print("✅ Download Complete.")


def upload_checkpoint(filepath):
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"checkpoints/{os.path.basename(filepath)}")
        blob.upload_from_filename(filepath)
        xm.master_print(f"☁️ Checkpoint uploaded to gs://{BUCKET_NAME}/checkpoints/")
        
    except Exception as e:
        xm.master_print(f"⚠️ Upload failed: {e}")

def _train_model(index):
    # 1. Setup Device
    device = xm.xla_device()
    xm.master_print(f"🚀 Core {index} ready on {device}")
    
    # 2. Download Data (Once on Master)
    if xm.is_master_ordinal():
        download_data_if_missing()
    # Wait for master to finish downloading
    xm.rendezvous('download_complete')

    # 3. Load Data
    dataset = load_from_disk(DATA_DIR)
    
    # Custom Collate for List -> Tensor
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        x = input_ids[:, :-1] # Inputs
        y = input_ids[:, 1:]  # Targets (Next token)
        return x, y

    # Distributed Sampler ensures each core gets different data
    train_sampler = DistributedSampler(
        dataset["train"],
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True, # Critical for TPU Speed
        num_workers=4
    )

    # 4. Initialize Model
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Wrap loader for TPU
    para_loader = pl.ParallelLoader(train_loader, [device])
    device_loader = para_loader.per_device_loader(device)

    # 5. Training Loop
    model.train()
    start_time = time.time()
    xm.master_print("🔥 Starting Training...")

    for step, (xb, yb) in enumerate(device_loader):
        if step >= MAX_ITERS: break

        # Forward
        logits, loss = model(xb, yb)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # TPU Step (Compiles graph & updates weights)
        xm.optimizer_step(optimizer)

        # Logging
        if step % 10 == 0:
            # xm.master_print ensures only one log is shown
            xm.master_print(f"Step {step} | Loss: {loss.item():.4f} | Time: {time.time()-start_time:.2f}s")
            start_time = time.time()

        # Save Checkpoint
        if step > 0 and step % SAVE_INTERVAL == 0:
            xm.master_print("💾 Saving Checkpoint...")
            xm.save(model.state_dict(), "mhla_model_checkpoint.pth")
            if xm.is_master_ordinal():
                upload_checkpoint("mhla_model_checkpoint.pth")

    xm.master_print("✅ Training Complete! Saving final model...")
    xm.save(model.state_dict(), "mhla_model_final.pth")
    
    if xm.is_master_ordinal():
        upload_checkpoint("mhla_model_final.pth")

if __name__ == "__main__":
    xmp.spawn(_train_model, args=(), nprocs=8, start_method='fork')
