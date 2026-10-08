import os 
import numpy as np
import tiktoken 
import urllib.request

SRC = "data/input.txt"
TRAIN_OUT = "data/train.bin"
VAL_OUT = "data/val.bin"

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


os.makedirs("data", exist_ok=True) 

if os.path.exists(SRC): 
    print(f"{SRC} already exists so skipping")
else: 
    print(f"downloading to {SRC}...")
    urllib.request.urlretrieve(URL, SRC)
    print(f"done, {os.path.getsize(SRC)} bytes")

if os.path.exists(TRAIN_OUT) and os.path.exists(VAL_OUT):
    print(f"Processed files already exist, skipping...")
else:
    text = open(SRC, 'r').read()
    enc = tiktoken.get_encoding('gpt2')

    data = enc.encode(text)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    np.array(train_data, dtype=np.uint16).tofile(TRAIN_OUT)
    np.array(val_data, dtype=np.uint16).tofile(VAL_OUT)    

    print(f"Size of train data: {os.path.getsize(TRAIN_OUT)} bytes")
    print(f"Size of validation data: {os.path.getsize(VAL_OUT)} bytes")
