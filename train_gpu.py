import torch
from datasets import load_dataset

from chat import m, model
from model import learning_rate, max_iters, eval_interval, eval_iters, device, block_size, batch_size

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

dataset = load_dataset("PHNG/chatmed-thaigpt1k-th", split='train', streaming=True)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()
    
    for split in ['train', 'val']:
    
        losses = torch.zeros(eval_iters)
    
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
    
        out[split] = losses.mean()
    
    model.train()
    return out


print("Starting training...")

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()