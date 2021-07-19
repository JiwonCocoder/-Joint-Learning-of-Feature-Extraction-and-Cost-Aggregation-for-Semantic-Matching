import torch
a = torch.randn(10, 4, 4)
a_mean = torch.mean(a)
a_max = torch.max(a)
print(a)
print(a_mean)
print(a_max)