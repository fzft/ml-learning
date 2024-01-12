from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, get_ds, run_validation

# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)


# load the pretrained weights
weights_file_path = get_weights_file_path(config, f'19')
state = torch.load(weights_file_path)
model.load_state_dict(state['model_state_dict'])

if __name__ == '__main__':
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                   lambda msg: print(msg), 0, None, num_examples=10)