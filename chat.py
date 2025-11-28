import os
import torch

from wordtokenizer import tokenizer
from model import GPTLanguageModel, device

def save_checkpoint(step, model, optimizer, loss, filename="checkpoint.pth"):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab_size': model.token_embedding_table.weight.shape[0] # Save vocab size for safety checks
    }
    torch.save(checkpoint, filename)
    print(f"--> Saved checkpoint to {filename}")

def load_checkpoint(filename, model, optimizer):
    if not os.path.exists(filename):
        print(f"--> No checkpoint found at {filename}, starting from scratch.")
        return 0 # start_iter

    print(f"--> Loading checkpoint from {filename}...")
    checkpoint = torch.load(filename, map_location=device)
    
    # Safety Check: Ensure vocab sizes match
    saved_vocab = checkpoint.get('vocab_size', None)
    current_vocab = model.token_embedding_table.weight.shape[0]
    if saved_vocab is not None and saved_vocab != current_vocab:
        raise ValueError(f"Vocab size mismatch! Saved: {saved_vocab}, Current: {current_vocab}. Did you retrain the tokenizer?")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"--> Resumed from step {step} with loss {loss:.4f}")
    return step


def stream_chat(model_instance: GPTLanguageModel, user_prompt: str) -> None:

    formatted_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    context_ids = tokenizer.encode(formatted_prompt).ids

    if not context_ids:
        context_ids = [tokenizer.token_to_id("<|pad|>")]

    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    stop_token_id = tokenizer.token_to_id("<|im_end|>")

    generated_ids = []

    for next_token_tensor in model_instance.generate_stream(context, max_new_tokens=5000, temperature=0.8):

        next_token_id = next_token_tensor[0].item()

        if next_token_id == stop_token_id:
            print(tokenizer.decode([next_token_id]))
            break
            
        generated_ids.append(next_token_id)
        text_chunk = tokenizer.decode([next_token_id])
        
        print(text_chunk, end='', flush=True)

    print()