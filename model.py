from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# classes go here 

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2

class Head(nn.Module): 
    # ONE head of attention 

    def  __init__(self, config, head_size):
        super().__init__() 
        self.head_size = head_size
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)    
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) # lower triangular matrix

    def forward(self,  x): 
        B, T, C = x.shape 
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module): 
    # multiple heads of self-attention in parallel 

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, config.n_embd // config.n_head) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate all head outputs
        out = self.proj(out) # project back to original embedding size
        out = self.dropout(out)
        return out

class FeedForward(nn.Module): 
    def __init__(self, config):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), 
            nn.ReLU(), 
            nn.Linear(4 * config.n_embd, config.n_embd), 
            nn.Dropout(config.dropout)      
        )

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
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
