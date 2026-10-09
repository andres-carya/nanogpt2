import argparse
import torch
import tiktoken 
from model import GPT 

#add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='out/gpt.pth',
                    help='checkpoint')
parser.add_argument('--prompt', type=str, default='ROMEO',
                    help='input string')
parser.add_argument('--tokens', type=int, default=200,
                    help='number of tokens to generate')
args = parser.parse_args()

#load 
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
model = GPT(ck['config'])
model.to(device)
model.load_state_dict(ck['model'])
model.eval()
    
prompt = tiktoken.get_encoding("gpt2").encode(args.prompt)
prompt = torch.tensor(prompt, dtype=torch.long, device=device)
prompt = prompt.view(1, -1)

with torch.no_grad():
    output = model.generate(prompt, args.tokens)
    dec = tiktoken.get_encoding("gpt2").decode(output[0].tolist())
    print(dec)  
