import torch
import threading

def train_fn():
    x = torch.ones(5, 5, requires_grad=True)
    # Forward
    y = (x + 3) * (x + 4) * 0.5
    
    # backward
    y.sum().backward()


# User write their own threading code to drive the train_fn
threads = []
for _ in range(10):
    p = threading.Thread(target=train_fn, args=())
    p.start()
    threads.append(p)

for p in threads:
    p.join()
