import torch
import torch.nn as nn
from torch.nn import functional as F

from wordtokenizer import tokenizer
from RoPE import RotaryPositionalEmbedding

batch_size = 32      # How many independent sequences will we process in parallel?
block_size = 64      # What is the maximum context length for predictions?
max_iters = 2000     # Total training iterations
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 768         # Embedding dimension
n_head = 4           # Number of attention heads
n_layer = 4          # Number of transformer blocks
kv_lora_rank = 64    # MLA: The dimension of the compressed latent vector
dropout = 0.2

vocab = tokenizer.get_vocab()
vocab_size = tokenizer.get_vocab_size()

class MultiHeadLatentAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
    
        super().__init__()
    
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Initialize RoPE
        self.rope = RotaryPositionalEmbedding(head_size, block_size)
        
        self.kv_down = nn.Linear(n_embd, kv_lora_rank, bias=False)
        self.kv_ln = nn.LayerNorm(kv_lora_rank)
        self.kv_up = nn.Linear(kv_lora_rank, 2 * num_heads * head_size, bias=False)
        self.q_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    
    def forward(self, x):
    
        B, T, C = x.shape
        
        # --- Query Generation ---
        q = self.q_proj(x) 
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, H, T, Head_Size)

        # --- Latent KV Generation ---
        compressed_kv = self.kv_down(x)
        compressed_kv = self.kv_ln(compressed_kv)
        kv = self.kv_up(compressed_kv)
        kv = kv.view(B, T, 2, self.num_heads, self.head_size)
        
        k = kv[:, :, 0, :, :].transpose(1, 2) # (B, H, T, Head_Size)
        v = kv[:, :, 1, :, :].transpose(1, 2) # (B, H, T, Head_Size)
        
        # --- Apply RoPE ---
        # Get cos and sin tables for current sequence length T
        cos, sin = self.rope(q)
        q = self.rope.apply_rotary_emb(q, cos, sin)
        k = self.rope.apply_rotary_emb(k, cos, sin)
        
        # --- Attention ---
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
    
        super().__init__()
    
        head_size = n_embd // n_head
    
        self.sa = MultiHeadLatentAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.token_embedding_table.weight = self.lm_head.weight

    
    def forward(self, idx, targets=None):
    
        B, T = idx.shape
        x = self.token_embedding_table(idx)
                
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    
    def generate(self, idx, max_new_tokens, temperature=1.0):
    
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    
        return idx


    def generate_stream(self, idx, max_new_tokens, temperature=1.0):

        """Yields one token at a time for streaming generation"""

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next

