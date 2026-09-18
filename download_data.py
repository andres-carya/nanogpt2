import os 
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = "data/input.txt"

os.makedirs("data", exist_ok=True) 

if os.path.exists(DEST): 
    print(f"{DEST} already exists so skipping")
else: 
    print(f"downloading to {DEST}...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done, {os.path.getsize(DEST)} bytes")