# gpu_benchmark.py
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on {device}")
start = time.time()

# Heavy operation
x = torch.randn(10000, 10000, device=device)
for _ in range(100):
    x = x @ x
torch.cuda.synchronize()

end = time.time()
print(f"Completed in {end - start:.2f} seconds")