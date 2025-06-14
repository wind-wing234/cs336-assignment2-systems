import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    model = ToyModel(in_features=5, out_features=2).cuda()

    x = torch.randn(3, 5).cuda()
    # 创建随机目标标签，用于交叉熵计算
    target = torch.randint(0, 2, (3,)).cuda()

    with torch.autocast("cuda", dtype=torch.float16):
        output = model(x)
        # 交叉熵loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)  # 交叉熵损失会使用高精度计算
        loss.backward()
        for param in model.parameters():
            a = param.grad # 梯度精度和参数保持一致
            print(a)

# model parameters: float32
# x: float32
# self.fc1(x): float16
# self.relu(self.fc1(x)): float16
# self.ln(x): float32
# self.fc2(x): float16

# output: float16
# loss: float32
# grad: float32
