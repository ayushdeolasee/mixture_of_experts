import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

# Device

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Hyperparameters
SEQ_LENGHT = 4096
VOCAB_SIZE = 50304
EMBEDDING_DIM = 1024
NUM_HEADS = 16
NUM_BLOCKS = 16
BATCH_SIZE = 128
NUM_EXPERTS = 64
TOP_K_EXPERTS = 8

# Data loader

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class data_loader:
    def __init__(self, B, T, split, data_root):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        data_root = data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        if (len(shards) > 0) == False:
            print(f"no shards found for split {split}")
        print(f"[green]found {len(shards)} shards for split {split}[/green]")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

# Rotary Position Embedding
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)
    def apply_rotary(self, x, seq_len):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = self.cos[:seq_len].unsqueeze(0).to(x.device)
        sin = self.sin[:seq_len].unsqueeze(0).to(x.device)
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated
    def forward(self, x):
        return self.apply_rotary(x, x.size(1))

# SwiGLU
class SwiGLU(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.linear1 = nn.Linear(input_dimension, 2 * hidden_dimension, bias=True)
        self.linear2 = nn.Linear(hidden_dimension, input_dimension, bias=True)
    def forward(self, x):
        combined = self.linear1(x)
        a, b = combined.chunk(2, dim=-1)
        swish = b * torch.sigmoid(b)
        output = self.linear2(swish * a)
        return output

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, input_shape, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(input_shape))
        self.b = nn.Parameter(torch.ones(input_shape))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = x / rms 
        output = (output * self.g) + self.b
        return output 

# Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, eps=1e-5):
        super().__init__()
        self.n_embd = n_embed
        self.n_head = n_head
        self.eps = eps 
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.alpha = nn.Parameter(torch.ones(self.n_head))
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        q_hat = q / (q_norm + self.eps)
        k_hat = k / (k_norm + self.eps)
        factor = self.alpha * math.sqrt(C // self.n_head)
        factor = factor.view(1, self.n_head, 1, 1)
        q_scaled = q_hat * factor
        y = F.scaled_dot_product_attention(q_scaled, k_hat, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

# Expert
class Expert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            SwiGLU(4 * embed_dim, 4 * embed_dim),
            nn.Linear(4 * embed_dim, embed_dim),
        )
    def forward(self, x): 
        return self.expert(x)

# Router
class Router(nn.Module):
    def __init__(self, num_experts, embed_dim):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.router = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_experts),
            nn.Softmax(dim=-1),
        )
    def forward(self, x): 
        return self.router(x)

# Block
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.RMSNorm = RMSNorm(self.embed_dim)
        self.MultiheadAttention = CausalSelfAttention(self.embed_dim, self.num_heads)
        self.router = Router(self.num_experts, self.embed_dim)
        self.experts = nn.ModuleList([Expert(self.embed_dim) for _ in range(self.num_experts)])
    def forward(self, x):
        x = x + self.MultiheadAttention(self.RMSNorm(x))
        routes = self.router(x)
        top8_probs, top8_indices = torch.topk(routes, k=8, dim=2)
        top8_probs = top8_probs / top8_probs.sum(dim=-1, keepdim=True)
        expert_output = torch.zeros_like(x)
        for k in range(8):
            expert_idx = top8_indices[:, :, k]
            prob = top8_probs[:, :, k]
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.sum() == 0:
                    continue
                x_selected = x[mask]
                expert_out = self.experts[expert_id](x_selected)
                prob_selected = prob[mask].unsqueeze(-1)
                weighted_out = expert_out * prob_selected
                expert_output[mask] += weighted_out
        x = x + expert_output
        return x

class Model(nn.Module):
    def __init__(self, seq_lenght, vocab_size, embed_dim):
        super().__init__()
        self.seq_lenght = seq_lenght
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embedding = RotaryPositionEmbedding(self.embed_dim, self.vocab_size)
        
        self.blocks = Block(self.embed_dim, self.num_heads)
        self.rmsnorm = RMSNorm(self.embed_dim)
        self.output_linear = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, x, layers=16):
        B, T = x.shape
        x = self.position_embedding(self.embedding(x))
        x += self.blocks(x)
        output = self.rmsnorm(x)
        output = self.output_linear(output)

        return output
