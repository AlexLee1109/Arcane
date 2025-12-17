import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from Arcane.kv_cache import KVCache

@dataclass
class GPTConfig:
    block_size: int = 1024   
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 6
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """Causal self-attention with Grouped Query Attention (GQA) and RoPE."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Precompute rotary embeddings up to max context
        self._build_rope_cache(config.block_size)

    def _build_rope_cache(self, max_pos):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        t = torch.arange(max_pos)
        freqs = torch.einsum("i,j->ij", t, inv_freq).repeat_interleave(2, dim=1)

        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def _apply_rope(self, x, pos):
        T = x.shape[2]
        cos = self.cos[..., pos:pos + T, :]
        sin = self.sin[..., pos:pos + T, :]

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(self, x, pos, kv_cache=None, layer_idx=None):
        B, T, _ = x.shape

        # Shape: [B, heads, T, head_dim]
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)

        # Append cached K/V for autoregressive inference
        use_cache = kv_cache is not None and layer_idx is not None
        if use_cache:
            past_kv = kv_cache.get(layer_idx)
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        # Full causal mask only needed during prefill
        is_causal = not use_cache or kv_cache.get(layer_idx) is None

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
        y = out.transpose(1, 2).contiguous().view(B, T, -1)

        if use_cache:
            kv_cache.update(layer_idx, (k, v))

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256 
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x, pos, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.ln_1(x), pos, kv_cache, layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.RMSNorm(config.n_embd, eps=1e-6),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"Model loaded: {total_params:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None):
        pos = 0 if kv_cache is None or kv_cache.is_empty() else kv_cache.get(0)[0].shape[2]
        x = self.transformer.wte(idx)

        for i, block in enumerate(self.transformer.h):
            x = block(x, pos, kv_cache, layer_idx=i)

        return self.lm_head(self.transformer.ln_f(x))

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_tokens,
        temperature=1.0,
        top_k=None,
        eos_tokens=None,
        min_tokens=0,
        kv_cache=None,
    ):
        device = next(self.parameters()).device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        kv_cache = kv_cache or KVCache(self.config.n_layer)

        # Prefill prompt into KV cache
        if input_ids.shape[1] > 0:
            self(input_ids, kv_cache)

        eos_set = {eos_tokens} if isinstance(eos_tokens, int) else set(eos_tokens or [])
        generated_count = 0

        for _ in range(max_tokens):
            logits = self(input_ids[:, -1:], kv_cache)[:, -1]

            if temperature != 1.0:
                logits /= temperature

            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

            next_token = torch.multinomial(
                torch.softmax(logits, dim=-1), 1
            ).item()

            generated_count += 1
            yield next_token

            # Append token for next-step conditioning
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device)], dim=1
            )

            if eos_set and next_token in eos_set and generated_count >= min_tokens:
                break

    def configure_optimizers(self, weight_decay, learning_rate):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith('.bias') or 'ln' in name or 'wte' in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return torch.optim.AdamW([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)