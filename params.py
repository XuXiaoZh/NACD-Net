import torch
from model_v3 import NoiseAwareDenoiserV3
#
# model = NoiseAwareDenoiserV3(z_dim=128, cond_len=400).to("cpu")
#
# # 只统计可训练参数
# n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"参数量: {n_params / 1e6:.4f} M")

import torch
import time

device = torch.device("cuda")
model  = NoiseAwareDenoiserV3(z_dim=128, cond_len=400).to(device).eval()

x      = torch.randn(1, 3, 6000).to(device)
z_cond = torch.randn(1, 3, 400).to(device)

# 预热（消除 CUDA 初始化开销）
with torch.no_grad():
    for _ in range(10):
        model(x, z_cond)

# 正式计时
starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)

N = 100
times = []
with torch.no_grad():
    for _ in range(N):
        starter.record()
        model(x, z_cond)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

print(f"推理时间（GPU）: 均值={sum(times)/N:.3f} ms  中位={sorted(times)[N//2]:.3f} ms")