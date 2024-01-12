from typing import List

import torch
from dataclasses import dataclass
from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)

@dataclass
class CacheInputMetadata:
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor

    # how many elements are cached per sequence
    cached_elements: torch.Tensor

    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]

class CacheView:

    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: CacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self.cache_k = torch.cat([self.cache_k, k], dim=1)
        self.cache_v = torch.cat([self.cache_v, v], dim=1)

    def interleave_kv(self, k: torch.Tensor, v: torch.Tensor):
        # k, v are (B, Seq_Len, n_kv_heads, head_dim)


    @property
    def prefill(self):
        return self.metadata.prefill



class RotatingBufferCache:

    def __init__(self, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):
        self.max_batch_size = max_batch_size
        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache_k = torch.empty(max_batch_size, sliding_window, n_kv_heads, head_dim)
        self.cache_v = torch.empty(max_batch_size, sliding_window, n_kv_heads, head_dim)
        self.kv_seqlens = None


    def init_kv_seqlens(self, batch_size:int):
        self.kv_seqlens = torch.zeros((batch_size,), dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> CacheInputMetadata:
        #

        if self.kv_seqlens is None:
            self.init_kv_seqlens(len(seqlens))
        seqpos = self.kv_seqlens.tolist()
        assert len(seqlens) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"

        # seqlens = [3,4,5]
        masks = [[x >= seqlen - self.sliding_window for x in range(seqlen)] for seqlen in seqlens]
        print("masks", masks)
        # masks = [[False, False, True], [False, True, True, True], [True, True, True, True, True]]
        to_cache_mask = torch.tensor(sum(masks, []), dtype=torch.bool)
        print("to_cache_mask", to_cache_mask)
        cached_elements = torch.tensor([sum(mask) for mask in masks], dtype=torch.long)
        print("cached_elements", cached_elements)
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)])
        print("positions", positions)
        batch_idx = torch.tensor(sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []), dtype=torch.long)
        print("batch_idx", batch_idx)

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window)
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in
                           zip(seqlens, self.kv_seqlens)]
            ).make_local_attention_from_bottomright(self.sliding_window)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist()
            )
        return CacheInputMetadata(
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=positions % self.sliding_window + batch_idx * self.sliding_window,
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens
        )


if __name__ == '__main__':
    cache = RotatingBufferCache(1, 3, 8, 64)
    cache.get_input_metadata([3, 5, 7])