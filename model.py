from dataclasses import dataclass
import math
import torch
import torch.nn as nn 
from torch.nn import functional as F 

# classes go here 

@dataclass
class GPTConfig:
    # GPT-2 124M geometry
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.2

class MultiHeadAttention(nn.Module):
    # all heads of self-attention in parallel, in one matmul

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # key, query, value for ALL heads at once (bias to match GPT-2)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        # (1, 1, block_size, block_size) so it broadcasts over batch and head dims
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))
                                          .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head # head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # one matmul -> 3 x (B, T, C)
        # split the channel dim into heads and move heads next to batch, so the
        # matmuls below are batched over (B, n_head) instead of looped per head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # (B, n_head, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # (B, n_head, T, hs)

        wei = q @ k.transpose(-2, -1) * hs**-0.5 # (B, n_head, T, T)
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v # (B, n_head, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble heads side by side
        out = self.proj(out) # project back to original embedding size
        out = self.dropout(out)
        return out

class FeedForward(nn.Module): 
    def __init__(self, config):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), 
            nn.GELU(approximate='tanh'), 
            nn.Linear(4 * config.n_embd, config.n_embd), 
            nn.Dropout(config.dropout)      
        )
        self.net[2].NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        return self.net(x)

class Block(nn.Module): 
    # Transformer block: communication followed by computation 

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x

class GPT(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT-2 has no bias here
        self.lm_head.weight = self.token_embedding_table.weight # tie weights
        self.apply(self._init_weights) # initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = std * (1 / math.sqrt(2 * self.config.n_layer))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size) this is just a matmul 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate (self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # crop context to block_size
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx
    

if __name__ == "__main__":
    config = GPTConfig()
    m = GPT(config)
    print(sum(p.numel() for p in m.parameters()), "parameters")

    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    logits, loss = m(x, x)          # targets=x is nonsense but exercises the loss path
    print(logits.shape, loss.item())
