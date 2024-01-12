from typing import Optional, List
import torch
import os
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Transformer, ModelArgs


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs,
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        prev_time = time.time()
        if device == "cuda":
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            chk_path = checkpoints[-1]
            print(f"Loading model from {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Model loaded in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        torch.cuda.empty_cache()
        model = Transformer(model_args).to(device)
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3} GB")

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Model loaded in {time.time() - prev_time:.2f}s")
        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: List[str], temperature: float = .6, top_p: float = .9,
                        max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        # Convert the prompts to tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in
                         prompts]
        # Make sure the batch size is not too big
        batch_size = len(prompt_tokens)

        assert batch_size <= self.model_args.max_batch_size, f"Batch size cannot be greater than {self.model_args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the max sequence length
        assert max_prompt_len <= self.model_args.max_seq_len, f"Prompt length cannot be greater than {self.model_args.max_seq_len}"
        total_len = min(self.model_args.max_seq_len, max_prompt_len + max_gen_len)

        # Create the list that will contain the generated tokens, along with the prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, v in enumerate(prompt_tokens):
            tokens[k, :len(v)] = torch.tensor(v, dtype=torch.long, device="cuda")

        eos_reached = torch.tensor([False] * batch_size, device="cuda")
        prompt_token_mask = tokens != pad_id
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model(tokens[:, cur_pos - 1:cur_pos], cur_pos).detach()
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = self._sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                next_token = next_token.reshape(-1)
                next_token = torch.where(prompt_token_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token
                # EOS is reached only if we found an EOS token for a padding position
                eos_reached |= (~prompt_token_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
                if all(eos_reached):
                    break
        out_tokens = []
        out_next = []
        for prompt_index, current_prompt in enumerate(tokens.tolist()):
            # Cut the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt:
                current_prompt = current_prompt[:current_prompt.index(self.tokenizer.eos_id())]
            out_tokens.append(current_prompt)
            out_next.append(self.tokenizer.decode(current_prompt))
        return out_tokens, out_next

    def _sample_top_p(self, probs, top_p):
        probs_sorted, indices = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sorted, dim=-1)
        mask = probs_sum - probs_sorted > top_p
        probs_sorted[mask] = 0
        probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sorted, num_samples=1)
        next_token = torch.gather(indices, -1, next_token)
        return next_token


if __name__ == '__main__':
    torch.manual_seed(0)

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Chinese company founded in Shanghai, it would",
        # Few shot promt
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Fangzhenfutao
        Decision: 
        """
    ]

    model_path = "/mnt/sda/llama2/llama-2-13b/"

    model = LLaMA.build(model_path, "tokenizer.model", True, 1024, len(prompts), device=device)

    # Inference
    out_tokens, out_next = model.text_completion(prompts, temperature=.6, top_p=.9, max_gen_len=100)
    assert len(out_tokens) == len(prompts)
    for prompt, tokens, next in zip(prompts, out_tokens, out_next):
        print(f"Prompt: {prompt}")
        # print(f"Tokens: {tokens}")
        print(f"Next: {next}")
        print("----")
