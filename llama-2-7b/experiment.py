from model import Transformer, ModelArgs
import json
import torch

from sentencepiece import SentencePieceProcessor
tokenizer = SentencePieceProcessor()
tokenizer.Load("tokenizer.model")

max_seq_len = 128
max_batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("params.json", "r") as f:
    params = json.loads(f.read())
    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        **params
    )

model_args.vocab_size = tokenizer.vocab_size()
model = Transformer(model_args)
names, submodules = model.count_parameters()
print(f"Total parameters: {names}, {submodules}")