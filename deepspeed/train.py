import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import inspect
from rich import print
import logging
from rich.logging import RichHandler
import wandb
import matplotlib.pyplot as plt
import deepspeed


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"[green]Using device: {device}[/green]")

BATCH_SIZE = 1 
MAX_SEQ_LENGTH = 1024 
WARMUP_STEPS = 750
MIN_LR = 3e-6
MAX_LR = 3e-4
VOCAB_SIZE = 50309
NUM_HEADS = 16
NUM_EXPERTS = 64
GRAD_ACCUM_STEPS = 4
EPOCHS = 10000
LEARNING_RATE = 3e-4
EMBED_DIM = 1024
CHECKPOINT_EPOCH = 10
TOP_K_EXPERTS = 8
NUM_BLOCKS = 16

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
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
        # assert len(shards) > 0, f"no shards found for split {split}"
        print(f"[green]found {len(shards)} shards for split {split}[/green]")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y
    
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_length=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_length).float()
        freqs = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('cos_cached', torch.cos(freqs), persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(self, x):
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        x_rotated = self.apply_rotary_pos_emb(x, cos, sin)
        return x_rotated
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        x_rotated = x_rotated.flatten(start_dim=-2)
        return x_rotated
    

class SwiGLU(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        # First linear layer outputs 2 * hidden_dimension for the gate and value
        self.linear1 = nn.Linear(input_dimension, 2 * hidden_dimension, bias=True)
        # Second linear layer takes hidden_dimension and outputs input_dimension
        self.linear2 = nn.Linear(hidden_dimension, input_dimension, bias=True)
    
    def forward(self, x): 
        combined = self.linear1(x)
        a, b = combined.chunk(2, dim=-1)
        swish = b * torch.sigmoid(b)
        output = self.linear2(swish * a)
        return output
    
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
    
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, max_seq_lenght, eps=1e-5):
        super().__init__()
        self.n_embd = n_embed
        self.n_head = n_head
        self.head_dim = self.n_embd // self.n_head
        self.max_seq_lenght = max_seq_lenght 
        self.eps = eps 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.alpha = nn.Parameter(torch.ones(self.n_head))
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.rope = RotaryPositionEmbedding(self.head_dim, self.max_seq_lenght) 
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, number of heads, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, numebr of heads, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, number_of_heads, T, head_size)
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        q_hat = q / (q_norm + self.eps)
        k_hat = k / (k_norm + self.eps)
        
        factor = self.alpha * math.sqrt(C // self.n_head)
        factor = factor.view(1, self.n_head, 1, 1)
        q_scaled = q_hat * factor
        q_scaled = self.rope(q_scaled)
        k_hat = self.rope(k_hat)
        y = F.scaled_dot_product_attention(q_scaled, k_hat, v, is_causal=True, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
    
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
    
class Router(nn.Module):
    def __init__(self, num_experts, embed_dim):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.linear = nn.Linear(self.embed_dim, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): 
        logits = self.linear(x)
        output = self.softmax(logits)
        return output, logits
    
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, max_seq_lenght):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_lenght = max_seq_lenght

        self.RMSNorm = RMSNorm(self.embed_dim)
        self.MultiheadAttention = CausalSelfAttention(self.embed_dim, self.num_heads, self.max_seq_lenght)
        self.router = Router(self.num_experts, self.embed_dim)
        self.experts = nn.ModuleList([Expert(self.embed_dim) for _ in range(self.num_experts)])


    def forward(self, x):
        x = x + self.MultiheadAttention(self.RMSNorm(x))
        routes, xj_logits = self.router(x)
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
        
        return x, top8_indices, top8_probs, xj_logits

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_lenght, num_heads, num_experts, num_blocks=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_lenght = max_seq_lenght
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.num_blocks = num_blocks

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim, dtype=torch.float32).to(device)
        
        self.blocks = nn.ModuleList([Block(self.embed_dim, self.num_heads, self.num_experts, self.max_seq_lenght) for _ in range(self.num_blocks)])
        self.rmsnorm = RMSNorm(self.embed_dim)
        self.output_linear = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)
        for block in self.blocks:
            x, top8_indicies, top8_probs, xj_logits = block(x)
        output = self.rmsnorm(x)
        output = self.output_linear(output)

        return output, top8_indicies, top8_probs, xj_logits

def load_balancing_loss(num_experts: int,
                        topk_probs: torch.Tensor,      # [B, T, K]
                        topk_indices: torch.Tensor,    # [B, T, K]
                        alpha: float = 0.01):
    B, T, K = topk_indices.shape
    tot_tokens = B * T

    # mask[b,t,k,e] == 1 if that (token,k) routes to expert e
    mask = (topk_indices.unsqueeze(-1) ==
            torch.arange(num_experts, device=topk_indices.device))

    # f_i  –– fraction of tokens routed to expert i
    tokens_per_expert = mask.any(dim=2).sum((0,1)).float()      # [E]
    f = tokens_per_expert / tot_tokens

    # P_i –– mean router prob mass arriving at expert i
    probs_per_expert = (topk_probs.unsqueeze(-1) *
                        mask.float()).sum((0,1,2))              # [E]
    P = probs_per_expert / tot_tokens

    lb_loss = alpha * num_experts * (f * P).sum()
    return lb_loss

def router_z_loss(router_logits, beta=0.001):
    loss = torch.logsumexp(router_logits, dim=-1) ** 2 
    loss = torch.mean(loss)
    return beta * loss

# Helper function for detailed expert utilization logging
def log_expert_utilization(top8_indices, top8_probs, num_experts, epoch):
    """
    Log detailed expert utilization metrics and visualizations to wandb
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate expert usage counts
    expert_counts = torch.bincount(top8_indices.view(-1), minlength=num_experts).float()
    expert_usage_percentage = (expert_counts / expert_counts.sum()) * 100
    
    # Create expert utilization histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Expert usage distribution
    ax1.bar(range(num_experts), expert_counts.cpu().numpy())
    ax1.set_xlabel('Expert ID')
    ax1.set_ylabel('Number of Tokens Routed')
    ax1.set_title(f'Expert Utilization Distribution (Epoch {epoch})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expert usage percentage
    ax2.bar(range(num_experts), expert_usage_percentage.cpu().numpy())
    ax2.set_xlabel('Expert ID')
    ax2.set_ylabel('Percentage of Total Tokens (%)')
    ax2.set_title(f'Expert Usage Percentage (Epoch {epoch})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({
        f"expert_utilization/distribution_epoch_{epoch}": wandb.Image(fig),
        "expert_utilization/expert_counts": wandb.Histogram(expert_counts.cpu().numpy()),
        "expert_utilization/routing_probs": wandb.Histogram(top8_probs.cpu().numpy()),
        "expert_utilization/std_dev": torch.std(expert_counts).item(),
        "expert_utilization/coefficient_of_variation": (torch.std(expert_counts) / torch.mean(expert_counts)).item(),
    })
    
    plt.close(fig)
    
    return expert_counts, expert_usage_percentage

def get_lr(it, warmup_steps, max_lr, max_steps, min_lr):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def train(batch_size, max_seq_lenght, vocab_size, num_heads, num_experts, grad_accum_steps, epochs, lr, device, embed_dim, checkpoint_epoch, warmup_steps, max_lr, max_steps, min_lr, project_name="mixture-of-experts", run_name=None):
    config = {
        "batch_size": batch_size,
        "max_seq_length": max_seq_lenght,
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "num_experts": num_experts,
        "grad_accum_steps": grad_accum_steps,
        "epochs": epochs,
        "learning_rate": lr,
        "device": device,
        "embed_dim": embed_dim,
        "top_k_experts": TOP_K_EXPERTS,
        "num_blocks": NUM_BLOCKS, 
        "warmup_steps": warmup_steps,
        "max_lr": max_lr,
        "max_steps": max_steps,
        "min_lr": min_lr
    }
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=["mixture-of-experts", "transformer", "moe"]
    )
    
    train_dataloader = DataLoader(B=batch_size, T=max_seq_lenght, split="train", data_root="/teamspace/studios/this_studio/data")
    val_dataloader = DataLoader(B=batch_size, T=max_seq_lenght, split="val", data_root="/teamspace/studios/this_studio/data")

    model = Model(vocab_size, embed_dim, max_seq_lenght, num_heads, num_experts).to(device)
    
    # Log model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })

    # DeepSpeed: initialize
    ds_config_path = os.path.join(os.path.dirname(__file__), "config.json")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # Watch model for gradient tracking
    wandb.watch(model, log="all", log_freq=100)
    
    for epoch in range(epochs):
        model_engine.train()
        model_engine.zero_grad()
        loss_acum = 0.0
        train_lb_loss_acum = 0.0
        train_rz_loss_acum = 0.0
        train_ce_loss_acum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output, top8_indicies, top8_probs, xj_logits = model_engine(x)
                train_lb_loss = load_balancing_loss(num_experts, top8_probs, top8_indicies)
                train_rz_loss = router_z_loss(xj_logits)
                train_ce_loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
                train_loss = train_ce_loss + train_lb_loss + train_rz_loss
            train_loss = train_loss / grad_accum_steps
            loss_acum += train_loss.detach()
            train_lb_loss_acum += train_lb_loss.detach() / grad_accum_steps
            train_rz_loss_acum += train_rz_loss.detach() / grad_accum_steps
            train_ce_loss_acum += train_ce_loss.detach() / grad_accum_steps
            model_engine.backward(train_loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_engine.parameters(), 1.0)
        model_engine.step()
        lr = get_lr(epoch, warmup_steps, max_lr, max_steps, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Validation
        model_engine.eval()
        with torch.no_grad():
            x, y = val_dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output, top8_indicies, top8_probs, xj_logits = model_engine(x)
                val_lb_loss = load_balancing_loss(num_experts, top8_probs, top8_indicies)
                val_rz_loss = router_z_loss(xj_logits)
                val_ce_loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
                val_loss = val_ce_loss + val_lb_loss + val_rz_loss
        if epoch % checkpoint_epoch == 0:
            model_engine.save_checkpoint("./weights", tag=f"model-{run_name}")
        # Calculate expert utilization metrics
        expert_counts = torch.bincount(top8_indicies.view(-1), minlength=num_experts).float()
        expert_utilization = (expert_counts > 0).sum().item() / num_experts
        expert_load_variance = torch.var(expert_counts).item()
        wandb.log({
            # Training losses
            "train/total_loss": loss_acum.item(),
            "train/cross_entropy_loss": train_ce_loss_acum.item(),
            "train/load_balancing_loss": train_lb_loss_acum.item(),
            "train/router_z_loss": train_rz_loss_acum.item(),
            # Validation losses
            "val/total_loss": val_loss.item(),
            "val/cross_entropy_loss": val_ce_loss.item(),
            "val/load_balancing_loss": val_lb_loss.item(),
            "val/router_z_loss": val_rz_loss.item(),
            # Expert utilization metrics
            "experts/utilization_rate": expert_utilization,
            "experts/load_variance": expert_load_variance,
            "experts/mean_routing_prob": top8_probs.mean().item(),
            "experts/max_routing_prob": top8_probs.max().item(),
            "experts/min_routing_prob": top8_probs.min().item(),
            # Training metrics
            "training/epoch": epoch,
            "training/learning_rate": lr,
            "training/gradient_norm": grad_norm.item(),
            # Perplexity
            "train/perplexity": torch.exp(train_ce_loss_acum).item(),
            "val/perplexity": torch.exp(val_ce_loss).item(),
        })
        print(f"[purple]Epoch[/purple]: {epoch}| [blue]Train Loss[/blue]: {loss_acum.item():.4f} | [magenta]Val Loss[/magenta]: {val_loss.item():.4f} | [green]Expert Util[/green]: {expert_utilization:.3f} | [bold turquoise4]lr[/bold turquoise4]: {lr}")
        if epoch % 10 == 0:
            try:
                log_expert_utilization(top8_indicies, top8_probs, num_experts, epoch)
            except Exception as e:
                print(f"[yellow]Warning: Could not log expert utilization visualization: {e}[/yellow]")
    wandb.finish()

train(
    batch_size=BATCH_SIZE,
    max_seq_lenght=MAX_SEQ_LENGTH,
    vocab_size=VOCAB_SIZE,
    num_heads=NUM_HEADS,
    num_experts=NUM_EXPERTS,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    min_lr=MIN_LR,
    max_lr=MAX_LR,
    max_steps=EPOCHS,
    device=device,
    embed_dim=EMBED_DIM,
    checkpoint_epoch=CHECKPOINT_EPOCH,
    project_name="mixture-of-experts-training",
    run_name=f"moe-{NUM_EXPERTS}experts-{NUM_HEADS}heads-{EMBED_DIM}dim"
)
