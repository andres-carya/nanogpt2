import argparse
import os 
import torch 
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end run (seconds) to prove the pipeline works')
args = parser.parse_args()

# train config 
batch_size = 64
max_steps = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4

# model config -- GPT-2 124M geometry
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

ckpt_path = 'out/gpt.pth'

if args.smoke:
    # cheap enough to run every time you touch the training loop
    batch_size = 8
    max_steps = 20
    eval_interval = 10
    eval_iters = 10
    block_size = 32
    n_layer = 2
    n_head = 2
    n_embd = 64
    ckpt_path = 'out/gpt_smoke.pth'  # don't clobber a real checkpoint
    print("[smoke] tiny config, 20 steps")

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1337)

# data 
text = open('data/input.txt', 'r').read() 
chars = sorted(set(text))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#model 
config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
model = GPT(config).to(device)
print(config)
print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - config.block_size, (batch_size,))
    x = torch.stack([d[i:i+config.block_size] for i in ix])
    y = torch.stack([d[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#train 
for step in range(max_steps):
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

os.makedirs('out', exist_ok=True)
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'step': max_steps}, ckpt_path)
print (f"Training complete. Model saved to {ckpt_path}")

