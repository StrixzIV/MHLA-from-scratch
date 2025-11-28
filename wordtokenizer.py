import torch

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = [
    "<|im_start|>",  # Start of a turn
    "<|im_end|>",    # End of a turn
    "<|pad|>"        # Padding for batching
]

trainer = trainers.BpeTrainer(
    vocab_size=100000,
    special_tokens=special_tokens
)

def train_tokenizer(text: str) -> None:
    tokenizer.train_from_iterator([text], trainer)


def encode(s: str) -> list[int]:
    return tokenizer.encode(s).ids


def decode(l: torch.Tensor) -> str:
    return tokenizer.decode(l)


def text2tensor(text: str) -> torch.Tensor:
    return torch.tensor(encode(text), dtype=torch.long)
