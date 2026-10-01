import argparse
import os 
import torch 
import tiktoken
import time
from model import GPT, GPTConfig

# command --smoke line 
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end run (seconds) to prove the pipeline works')
parser.add_argument('--steps', type=int, default=None)
args = parser.parse_args()

# train config 
batch_size = 6
max_steps = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
log_interval = 10

# model geometry lives in model.py's GPTConfig -- override it only for smoke
model_overrides = {}

ckpt_path = 'out/gpt.pth'

if args.smoke:
    # cheap enough to run every time you touch the training loop
    batch_size = 8
    max_steps = 20
    eval_interval = 10
    eval_iters = 10
    model_overrides = dict(block_size=32, n_layer=2, n_head=2, n_embd=64)
    ckpt_path = 'out/gpt_smoke.pth'  # don't clobber a real checkpoint
    print("[smoke] tiny config, 20 steps")

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1337)

# data 
text = open('data/input.txt', 'r').read() 
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#model 
config = GPTConfig(vocab_size=vocab_size, **model_overrides)
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

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
        

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if step % log_interval == 0 or step == max_steps - 1:
        dt = t1 - t0
        tokens = config.block_size * batch_size
        tps = tokens / dt
        print(f"step {step}: loss {loss.item():.4f}, tps {tps:.2f}")

os.makedirs('out', exist_ok=True)
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'step': max_steps}, ckpt_path)
print (f"Training complete. Model saved to {ckpt_path}")

