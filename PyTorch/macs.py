import torch
from torchprofile import profile_macs
from model import LYT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LYT().to(device)

input_tensor = torch.randn(1, 3, 256, 256).to(device) 

macs = profile_macs(model, input_tensor)
flops = macs * 2
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

tflops = flops / (1024*1024*1024)

print(f"Model FLOPs (G): {tflops} G")
print(f"Model FLOPs (M): {tflops*1024} M")

print(f"Model MACs (G): {macs / (1024*1024*1024)} G")

print(f"Model params (M): {num_params / 1e6}")
print(f"Model params: {num_params}")