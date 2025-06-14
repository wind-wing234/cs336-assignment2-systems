import torch

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32) # 0.01000000
print(f"{s:.8f}") # 10.00013351

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16) # 0.01000214
print(f"{s:.8f}") # 9.95312500

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16) # 0.01000214 被转化为float32再和s相加
print(f"{s:.8f}") # 10.00213623

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(f"{s:.8f}") # 10.00213623

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(f"{s:.8f}") # 9.95312500

s = torch.tensor(0,dtype=torch.float16)
s = s.to(dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
s = s.to(dtype=torch.float16)
print(f"{s:.8f}") # 10.00000000