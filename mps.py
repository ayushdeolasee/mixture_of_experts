import mlx
import mlx.core as mx
import mlx.core.metal as metal
import numpy as np
import os
import mlx.nn as nn
import wandb
import matplotlib.pyplot as plt
import math
from mlx.core import value_and_grad
import mlx.optimizers as optim
import mlx.utils

BATCH_SIZE = 2  
MAX_SEQ_LENGTH = 512 
WARMUP_STEPS = 750
MIN_LR = 3e-6
MAX_LR = 3e-4
VOCAB_SIZE = 25000 
NUM_HEADS = 8 
NUM_EXPERTS = 32 
GRAD_ACCUM_STEPS = 8 
EPOCHS = 10000
LEARNING_RATE = 3e-4
EMBED_DIM = 512 
CHECKPOINT_EPOCH = 10
TOP_K_EXPERTS = 8
NUM_BLOCKS = 8  

# GPU profiling configuration
ENABLE_GPU_PROFILING = True
GPU_TRACE_FILE = "mlx_moe_training.gputrace"
PROFILE_EVERY_N_EPOCHS = 50  # Profile every N epochs
PROFILE_DURATION_EPOCHS = 1  # How many epochs to profile

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return mx.array(npt, dtype=mx.int32)

class DataLoader:
    def __init__(self, B, T, split, data_root):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s])
        self.shards = shards
        if not self.shards:
            print(f"Warning: no shards found for split {split}")
        else:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        if not self.shards:
            self.tokens = mx.array([], dtype=mx.int32)
            self.current_position = 0
            return
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        if not self.shards:
            return mx.zeros((B, T), dtype=mx.int32), mx.zeros((B, T), dtype=mx.int32)
        
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)
        
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

# --- Model Components (Corrected) ---

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_length=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_seq_length, dtype=mx.float32)
        freqs = mx.einsum("i,j->ij", t, inv_freq)
        self.cos_cached = mx.cos(freqs)
        self.sin_cached = mx.sin(freqs)

    def __call__(self, x):
        seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
        sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
        
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        
        x_rotated = mx.stack([x1_rotated, x2_rotated], axis=-1)
        return x_rotated.reshape(x.shape)

class SwiGLU(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.linear1 = nn.Linear(input_dimension, 2 * hidden_dimension, bias=True)
        self.linear2 = nn.Linear(hidden_dimension, input_dimension, bias=True)

    def __call__(self, x):
        a, b = mx.split(self.linear1(x), 2, axis=-1)
        return self.linear2(a * nn.silu(b))

class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super().__init__()
        self.g = mx.ones((dims,))
        self.b = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.g + self.b

def scaled_dot_product_attention(q, k, v, is_causal=True):
    # q, k, v: (B, n_head, T, head_dim)
    d_k = q.shape[-1]
    attn_scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(d_k)  # (B, n_head, T, T)
    if is_causal:
        seq_len = attn_scores.shape[-1]
        mask = mx.tril(mx.ones((seq_len, seq_len), dtype=attn_scores.dtype))
        mask = mask[None, None, :, :]  # (1, 1, T, T)
        attn_scores = mx.where(mask == 0, float('-inf'), attn_scores)
    attn_weights = nn.softmax(attn_scores, axis=-1)
    return mx.matmul(attn_weights, v)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, max_seq_lenght, eps=1e-5):
        super().__init__()
        self.n_embd = n_embed
        self.n_head = n_head
        self.head_dim = self.n_embd // self.n_head
        self.eps = eps
        
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_lenght)
        self.alpha = mx.ones(self.n_head)

    def __call__(self, x):
        B, T, C = x.shape
        q, k, v = mx.split(self.c_attn(x), 3, axis=-1)
        
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q_norm = mx.linalg.norm(q, ord=2, axis=-1, keepdims=True)
        k_norm = mx.linalg.norm(k, ord=2, axis=-1, keepdims=True)

        q_hat = q / (q_norm + self.eps)
        k_hat = k / (k_norm + self.eps)
        
        factor = self.alpha * math.sqrt(self.head_dim)
        q_scaled = q_hat * factor.reshape(1, self.n_head, 1, 1)
        
        q_scaled = self.rope(q_scaled)
        k_hat = self.rope(k_hat)

        y = scaled_dot_product_attention(q_scaled, k_hat, v, is_causal=True)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)

class Expert(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            SwiGLU(4 * embed_dim, 4 * embed_dim),
            nn.Linear(4 * embed_dim, embed_dim),
        )
    def __call__(self, x):
        return self.expert(x)

class Router(nn.Module):
    def __init__(self, num_experts, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_experts)

    def __call__(self, x):
        logits = self.linear(x)
        return nn.softmax(logits, axis=-1), logits

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, max_seq_lenght):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_lenght)
        self.rmsnorm = RMSNorm(embed_dim)
        self.router = Router(num_experts, embed_dim)
        self.experts = [Expert(embed_dim) for _ in range(num_experts)]
        self.num_experts = num_experts

    def __call__(self, x):
        x = x + self.attn(self.rmsnorm(x))
        
        routes, xj_logits = self.router(x)
        topk_probs, topk_indices = mx.top_k(routes, k=TOP_K_EXPERTS, axis=-1)
        topk_probs = topk_probs / mx.sum(topk_probs, axis=-1, keepdims=True)
        
        expert_outputs = mx.stack([expert(x) for expert in self.experts], axis=0)
        
        one_hot_indices = nn.one_hot(topk_indices, num_classes=self.num_experts)
        prob_dist = mx.sum(topk_probs[..., None] * one_hot_indices, axis=2)
        
        prob_dist_reshaped = prob_dist.transpose(2, 0, 1)[..., None]
        
        weighted_expert_outputs = expert_outputs * prob_dist_reshaped
        final_expert_output = mx.sum(weighted_expert_outputs, axis=0)
        
        x = x + final_expert_output
        return x, topk_indices, topk_probs, xj_logits

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_lenght, num_heads, num_experts, num_blocks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = [
            Block(embed_dim, num_heads, num_experts, max_seq_lenght) for _ in range(num_blocks)
        ]
        self.rmsnorm = RMSNorm(embed_dim)
        self.output_linear = nn.Linear(embed_dim, vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x, top8_indices, top8_probs, xj_logits = block(x)
        output = self.rmsnorm(x)
        return self.output_linear(output), top8_indices, top8_probs, xj_logits

# --- Loss Functions ---
def load_balancing_loss(num_experts, topk_probs, topk_indices, alpha=0.01):
    B, T, K = topk_indices.shape
    mask = nn.one_hot(topk_indices, num_classes=num_experts)
    f = mx.sum(mask.any(axis=2), axis=(0,1)).astype(mx.float32) / (B * T)
    P = mx.sum(topk_probs[..., None] * mask, axis=(0,1,2)) / (B * T)
    return alpha * num_experts * mx.sum(f * P)

def router_z_loss(router_logits, beta=0.001):
    return beta * mx.mean(mx.logsumexp(router_logits, axis=-1) ** 2)

def cross_entropy_loss(logits, targets):
    return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))

# --- Training ---
def get_lr(it, warmup_steps, max_lr, max_steps, min_lr):
    if it < warmup_steps: return max_lr * (it + 1) / warmup_steps
    if it > max_steps: return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train():
    wandb.init(project="mixture-of-experts-training", name=f"moe-mlx-{NUM_EXPERTS}e-{NUM_HEADS}h-{EMBED_DIM}d", config={
        "batch_size": BATCH_SIZE, "max_seq_length": MAX_SEQ_LENGTH, "vocab_size": VOCAB_SIZE,
        "num_heads": NUM_HEADS, "num_experts": NUM_EXPERTS, "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "embed_dim": EMBED_DIM, "top_k_experts": TOP_K_EXPERTS,
        "num_blocks": NUM_BLOCKS, "warmup_steps": WARMUP_STEPS, "max_lr": MAX_LR, "max_steps": EPOCHS, "min_lr": MIN_LR,
        "gpu_profiling_enabled": ENABLE_GPU_PROFILING
    })

    train_loader = DataLoader(B=BATCH_SIZE, T=MAX_SEQ_LENGTH, split="train", data_root="./data")
    val_loader = DataLoader(B=BATCH_SIZE, T=MAX_SEQ_LENGTH, split="val", data_root="./data")
    
    model = Model(VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LENGTH, NUM_HEADS, NUM_EXPERTS, NUM_BLOCKS)
    mx.eval(model.parameters())
    
    total_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    wandb.config.update({"total_parameters": total_params, "trainable_parameters": total_params})
    
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        output, top8_indices, top8_probs, xj_logits = model(x)
        ce_loss = cross_entropy_loss(output, y)
        lb_loss = load_balancing_loss(NUM_EXPERTS, top8_probs, top8_indices)
        rz_loss = router_z_loss(xj_logits)
        return ce_loss + lb_loss + rz_loss  # Only return the loss

    grad_fn = value_and_grad(loss_fn)

    # Initial warmup to ensure everything is compiled before profiling
    print("Warming up model...")
    x_warmup, y_warmup = train_loader.next_batch()
    loss, _ = grad_fn(model, x_warmup, y_warmup)
    mx.eval(model.parameters())
    print("Warmup complete.")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Start GPU profiling if enabled and at the right epoch
        profiling_active = False
        if ENABLE_GPU_PROFILING and epoch % PROFILE_EVERY_N_EPOCHS == 0:
            trace_file = f"{GPU_TRACE_FILE}_epoch_{epoch}.gputrace"
            # Remove existing trace file if it exists
            if os.path.exists(trace_file):
                os.remove(trace_file)
            
            print(f"Starting GPU capture for epoch {epoch} -> {trace_file}")
            print("Note: Make sure to run with MTL_CAPTURE_ENABLED=1 environment variable")
            try:
                metal.start_capture(trace_file)
                profiling_active = True
            except Exception as e:
                print(f"Failed to start GPU capture: {e}")
                print("Continuing without GPU profiling...")
        
        # Accumulation loop
        accumulated_grads = mlx.utils.tree_map(lambda p: mx.zeros_like(p), model.parameters())
        total_loss, total_ce, total_lb, total_rz = 0.0, 0.0, 0.0, 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.next_batch()
            loss, grads = grad_fn(model, x, y)
            # Recompute aux losses for logging
            output, top8_indices, top8_probs, xj_logits = model(x)
            ce = cross_entropy_loss(output, y)
            lb = load_balancing_loss(NUM_EXPERTS, top8_probs, top8_indices)
            rz = router_z_loss(xj_logits)
            accumulated_grads = mlx.utils.tree_map(lambda acc, new: acc + new, accumulated_grads, grads)
            total_loss += loss.item()
            total_ce += ce.item()
            total_lb += lb.item()
            total_rz += rz.item()

        # Update model
        avg_grads = mlx.utils.tree_map(lambda g: g / GRAD_ACCUM_STEPS, accumulated_grads)
        optimizer.update(model, avg_grads)
        
        lr = get_lr(epoch, WARMUP_STEPS, MAX_LR, EPOCHS, MIN_LR)
        optimizer.learning_rate = lr

        # Validation
        model.eval()
        x_val, y_val = val_loader.next_batch()
        val_loss = loss_fn(model, x_val, y_val)
        output, val_indices, val_probs, xj_logits = model(x_val)
        val_ce = cross_entropy_loss(output, y_val)
        val_lb = load_balancing_loss(NUM_EXPERTS, val_probs, val_indices)
        val_rz = router_z_loss(xj_logits)
        
        # Stop GPU profiling if it was active
        if profiling_active:
            try:
                metal.stop_capture()
                print(f"GPU capture completed for epoch {epoch}")
                wandb.log({"gpu_profiling/trace_file": trace_file, "gpu_profiling/epoch": epoch})
            except Exception as e:
                print(f"Failed to stop GPU capture: {e}")
        
        # Logging
        wandb.log({
            "train/total_loss": total_loss / GRAD_ACCUM_STEPS, "train/cross_entropy_loss": total_ce / GRAD_ACCUM_STEPS,
            "train/load_balancing_loss": total_lb / GRAD_ACCUM_STEPS, "train/router_z_loss": total_rz / GRAD_ACCUM_STEPS,
            "val/total_loss": val_loss.item(), "val/cross_entropy_loss": val_ce.item(),
            "val/load_balancing_loss": val_lb.item(), "val/router_z_loss": val_rz.item(),
            "training/epoch": epoch, "training/learning_rate": lr,
            "val/perplexity": math.exp(val_ce.item())
        })
        
        print(f"Epoch: {epoch}| Train Loss: {total_loss/GRAD_ACCUM_STEPS:.4f} | Val Loss: {val_loss.item():.4f} | LR: {lr:.6f}")

        if epoch % CHECKPOINT_EPOCH == 0 and epoch > 0:
            model.save_weights(f"./weights/model_mlx_{epoch}.npz")

if __name__ == "__main__":
    # Print GPU profiling instructions
    if ENABLE_GPU_PROFILING:
        print("=" * 60)
        print("GPU PROFILING ENABLED")
        print("=" * 60)
        print("To capture GPU traces, run this script with:")
        print("MTL_CAPTURE_ENABLED=1 python mps.py")
        print("")
        print("Generated .gputrace files can be opened in Xcode:")
        print("1. Open Xcode")
        print("2. Go to Window > Developer Tools > Instruments")
        print("3. Choose 'Metal System Trace' or open the .gputrace file directly")
        print("4. Analyze GPU performance, dependencies, and bottlenecks")
        print("=" * 60)
        print("")
    
    train()
