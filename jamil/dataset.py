import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.seq = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src = src_target_pair["translation"][self.src_lang]
        tgt = src_target_pair["translation"][self.tgt_lang]

        # tokenize the source and target sentences
        enc_input_token = self.tokenizer_src.encode(src).ids
        dec_input_token = self.tokenizer_tgt.encode(tgt).ids

        enc_num_padding_tokens = self.seq - len(enc_input_token) - 2
        dec_num_padding_tokens = self.seq - len(dec_input_token) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("The sequence length is too short")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # decoder input is the target sentence with an extra [SOS] token at the beginning
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # label is the target sentence with an extra [EOS] token at the end
        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0

        )


        assert encoder_input.shape[0] == self.seq
        assert decoder_input.shape[0] == self.seq
        assert label.size(0) == self.seq

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),  # (1, seq, seq)
            "label": label,
            "src_text": src,
            "tgt_text": tgt
        }

def causal_mask(seq_len):
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int)
    return mask == 0