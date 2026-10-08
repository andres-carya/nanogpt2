import argparse
import math
import os 
import torch 
import tiktoken
import time
import numpy as np
import wandb
from model import GPT, GPTConfig

# command --smoke line 
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true',
                    help='tiny end-to-end run (seconds) to prove the pipeline works')
parser.add_argument('--steps', type=int, default=None,
                    help='override max_steps')
parser.add_argument('--batch-size', type=int, default=None,
                    help='override batch_size, e.g. --batch-size 32 when hunting for the memory limit')
parser.add_argument('--plot-lr', action='store_true',
                    help='draw the LR schedule and exit, without training')
parser.add_argument('--compile', action='store_true',
                    help='compile the model')
parser.add_argument('--wandb', action='store_true',
                    help='log training metrics to Weights & Biases')
args = parser.parse_args()

# train config 
batch_size = 6
max_steps = 5000
eval_interval = 500
checkpoint_interval = 500
eval_iters = 200
learning_rate = 3e-4
weight_decay = 0.1
betas = (0.9, 0.95)
log_interval = 10
grad_clip = 1.0
grad_accumulation_steps = 1
max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_frac = 0.05

# model geometry lives in model.py's GPTConfig -- override it only for smoke
model_overrides = {}

ckpt_path = 'out/gpt.pth'

if args.smoke:
    # cheap enough to run every time you touch the training loop
    batch_size = 6
    max_steps = 20
    eval_interval = 10
    eval_iters = 10
    model_overrides = dict(block_size=32, n_layer=2, n_head=2, n_embd=64)
    ckpt_path = 'out/gpt_smoke.pth'  # don't clobber a real checkpoint
    print("[smoke] tiny config, 20 steps")

if args.steps is not None:
    max_steps = args.steps

if args.batch_size is not None:
    batch_size = args.batch_size



device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1337)

# data -- pre-tokenised by prepare_data.py, memory-mapped so the dataset
# never has to fit in RAM
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/val.bin', dtype=np.uint16, mode='r')

#model 
config = GPTConfig(vocab_size=vocab_size, **model_overrides)
model = GPT(config).to(device)

#reference for compile 
raw_model = model

#compile
if args.compile:
    model = torch.compile(model)

print(config)
print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

if args.wandb:
    wandb.init(project="gpt-train", config={
        "batch_size": batch_size,
        "max_steps": max_steps,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "warmup_frac": warmup_frac,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "grad_accumulation_steps": grad_accumulation_steps,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "block_size": config.block_size,
    })

#build list of parameters 

# get all the parameters that require gradients
params = [p for p in model.parameters() if p.requires_grad]

# put into groups 
decay_params = [p for p in params if p.dim() >= 2]
print('len(decay_params):', len(decay_params))
print('parameters:  ', sum(p.numel() for p in decay_params))

non_decay_params = [p for p in params if p.dim() < 2]
print('len(non_decay_params):', len(non_decay_params))
print('parameters:  ', sum(p.numel() for p in non_decay_params))

# create the optimizer
optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': non_decay_params, 'weight_decay': 0.0}
], lr=learning_rate, betas=betas, eps=1e-8)


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - config.block_size, (batch_size,))
    # uint16 on disk -> int64 for the embedding lookup, one batch at a time
    x = torch.stack([torch.from_numpy(d[i:i+config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+config.block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.detach()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(step):
    warmup_steps = int(warmup_frac * max_steps)
    if step < warmup_steps:
        return max_lr * (step + 1)/ warmup_steps
    elif step > max_steps: 
        return min_lr
    else: 
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

def plot_lr(get_lr, max_steps, warmup_steps, width=72, height=18):
    # sample the schedule at `width` evenly spaced steps and draw it as blocks
    xs = [round(i * (max_steps - 1) / (width - 1)) for i in range(width)]
    ys = [get_lr(x) for x in xs]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    blocks = ' ▁▂▃▄▅▆▇█'

    print(f"\n  LR schedule   peak {hi:.2e}   floor {lo:.2e}   "
          f"warmup {warmup_steps}   total {max_steps}\n")
    for row in range(height, 0, -1):
        line = ''
        for y in ys:
            frac = (y - lo) / span * height
            if frac >= row:
                line += '█'
            elif frac > row - 1:
                line += blocks[int((frac - (row - 1)) * 8)]
            else:
                line += ' '
        print(f"  {lo + span * (row - 0.5) / height:8.2e} │{line}")
    print(f"  {'':8s} └{'─' * width}")

    axis = [' '] * width
    for t in [0, max_steps // 4, max_steps // 2, 3 * max_steps // 4]:
        col = round(t / (max_steps - 1) * (width - 1))
        for j, ch in enumerate(str(t)):
            if col + j < width:
                axis[col + j] = ch
    last = str(max_steps - 1)
    for j, ch in enumerate(last):
        axis[width - len(last) + j] = ch
    print(f"  {'':8s}  {''.join(axis)}")
    print(f"\n  peak {hi:.3e} around step {xs[ys.index(hi)]}   "
          f"ends at {ys[-1]:.3e} = {ys[-1]/hi*100:.0f}% of peak\n")

if args.plot_lr:
    plot_lr(get_lr, max_steps, int(warmup_frac * max_steps))
    raise SystemExit

os.makedirs('out', exist_ok=True)

#train
for step in range(max_steps):
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(step)
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if args.wandb:  
            wandb.log({'train_loss': losses['train'], 'val_loss': losses['val']}, step=step)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
  
    #Gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro in range(grad_accumulation_steps):
        xb, yb = get_batch('train')
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        loss = loss / grad_accumulation_steps
        loss_accum += loss.item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if step % log_interval == 0 or step == max_steps - 1:
        dt = t1 - t0
        tokens = config.block_size * batch_size * grad_accumulation_steps
        tps = tokens / dt
        print(f"step {step}: loss {loss_accum:.4f}, tps {tps:.2f}, norm {norm.item():.4f}")
        if args.wandb:
            wandb.log({'lr': get_lr(step), 'loss': loss_accum, 'dt': dt, 'tps': tps, 'norm': norm.item()}, step=step)

    if step % checkpoint_interval == 0 or step == max_steps - 1:
        torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'step': step}, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

print (f"Training complete.")
if args.wandb:
    wandb.finish()
