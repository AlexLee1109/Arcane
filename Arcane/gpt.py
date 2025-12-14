import math
from dataclasses import dataclass

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
    """Multi-head causal attention with Grouped Query Attention (GQA) and RoPE"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V, and output projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Precompute RoPE rotation matrices
        self._build_rope_cache(config.block_size)

    def _build_rope_cache(self, max_pos):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        t = torch.arange(max_pos)
        freqs = torch.outer(t, inv_freq).repeat_interleave(2, dim=-1)

        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def _apply_rope(self, x, pos):
        B, nh, T, hd = x.shape
        cos = self.cos[..., pos:pos + T, :]
        sin = self.sin[..., pos:pos + T, :]

        x1, x2 = x[..., :hd//2], x[..., hd//2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    def forward(self, x, pos, cache=None):
        B, T, C = x.shape

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)

        if cache:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None), enable_gqa=True)

        y = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), {'k': k, 'v': v} 


class MLP(nn.Module):
    """SwiGLU-style feed-forward network (Llama-inspired)"""
    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = (hidden + 255) // 256 * 256
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Single transformer block with pre-norm, attention, and MLP"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x, pos, cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), pos, cache)
        x = x + attn_out                  
        x = x + self.mlp(self.ln_2(x))   
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),    
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  
            ln_f=nn.RMSNorm(config.n_embd, eps=1e-6),    
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layer))

        params_m = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"Model: {params_m:.2f}M params")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, kv_cache=None):
        # Determine current position in sequence
        pos = 0 if not kv_cache or kv_cache.is_empty() else kv_cache.get(0)['k'].shape[2]

        x = self.transformer.wte(idx)
        cache = kv_cache or KVCache(self.config.n_layer)

        for i, block in enumerate(self.transformer.h):
            x, new_cache = block(x, pos, kv_cache.get(i) if kv_cache else None)
            cache.update(i, new_cache)

        logits = self.lm_head(self.transformer.ln_f(x))
        return logits, cache

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, eos_tokens=None, min_tokens=0, kv_cache=None):
        """Fast autoregressive generation with KV caching and sampling"""
        device = next(self.parameters()).device
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Prefill: process prompt once with caching
        kv_cache = kv_cache or KVCache(self.config.n_layer)
        if ids.shape[1]:
            _, kv_cache = self(ids, kv_cache)

        for _ in range(max_tokens):
            logits, kv_cache = self(ids[:, -1:], kv_cache)
            logits = logits[:, -1]

            if temperature != 1.0:
                logits /= temperature

            # Top-k filtering
            if top_k and top_k > 0:
                v = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < v] = -float('Inf')

            # Sample next token
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            token = next_token.item()
            yield token

            # Append to sequence
            ids = torch.cat([ids, next_token], dim=1)

            # Early stopping on EOS tokens after minimum length
            if eos_tokens and (len(ids[0]) - len(tokens)) >= min_tokens and token in eos_tokens:
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