import torch

from tokenizers import Tokenizer

import chat
import model

from chat import stream_chat

tokenizer = Tokenizer.from_file("./tokenizer.json")

chat.tokenizer = tokenizer
model.vocab_size = tokenizer.get_vocab_size()

m = model.GPTLanguageModel()

state_dict = torch.load('./models/best_model.pth')
m.load_state_dict(state_dict)

m = m.to(model.device)
m.eval()

while True:
    stream_chat(m, input('<user>: '))
