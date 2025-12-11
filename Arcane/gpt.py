import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 6
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        # Q: full heads, KV: shared heads (GQA)
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # === PRECOMPUTE ROPE CACHE ===
        self._build_rope_cache()

    def _build_rope_cache(self):
        # (block_size, head_dim//2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(self.block_size, dtype=torch.float32)  # (block_size,)
        freqs = torch.outer(t, inv_freq)                        # (block_size, head_dim//2)

        # Expand to full head_dim: [a, b] → [a, a, b, b]
        freqs = freqs.repeat_interleave(2, dim=-1)              # (block_size, head_dim)

        # Precompute cos and sin: (1, 1, block_size, head_dim)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        # Register as buffers (not saved in state_dict if persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _apply_rope(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, nh, T, hd = x.shape
        assert start_pos + T <= self.block_size, f"Sequence length exceeds block_size"

        # Slice precomputed cache: (1, 1, T, hd)
        cos = self.rope_cos[:, :, start_pos:start_pos + T, :]
        sin = self.rope_sin[:, :, start_pos:start_pos + T, :]

        # Split and rotate
        x1 = x[..., : hd // 2]
        x2 = x[..., hd // 2:]
        x_rot = torch.cat([-x2, x1], dim=-1)  # (B, nh, T, hd)

        return x * cos + x_rot * sin

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # (B, nh, T, hd)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv, T, hd)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv, T, hd)

        q = self._apply_rope(q, start_pos)
        k = self._apply_rope(k, start_pos)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        y = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = (hidden_dim + 256 - 1) // 256 * 256
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.w1(x), inplace=True) * self.w2(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd, eps=1e-6),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Model: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M params")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def generate(
        self,
        tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_tokens: Optional[set] = None,
        min_tokens: int = 0,
    ):
        """
        Generate tokens and stop early once a punctuation token is produced
        and the minimum generation threshold is met.

        Args:
            tokens: initial prompt token list
            max_tokens: max new tokens
            temperature: sampling temp
            top_k: top-k sample limit
            eos_tokens: set of punctuation token IDs (e.g., {token_id('.'), token_id('?'), token_id('!')})
            min_tokens: min # of generated tokens before EOS stopping allowed
        """
        assert isinstance(tokens, list), "Input tokens must be a list of integers"
        device = next(self.parameters()).device

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = 0

        for _ in range(max_tokens):
            logits, _ = self(ids)
            logits = logits[:, -1, :]  # last step

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sampling
            probs = F.softmax(logits, dim=-1)
            if temperature > 0:
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = torch.argmax(probs, dim=-1, keepdim=True)

            next_token = next_ids.item()
            ids = torch.cat((ids, next_ids), dim=1)
            generated += 1

            yield next_token

            # Stop if punctuation and threshold met
            if eos_tokens is not None and generated >= min_tokens:
                if next_token in eos_tokens:
                    break

    def configure_optimizers(self, weight_decay: float, learning_rate: float):
        decay, nodecay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or any(x in name for x in ["ln", "norm", "wte"]):
                nodecay.append(p)
            else:
                decay.append(p)

        return torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0}
        ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)