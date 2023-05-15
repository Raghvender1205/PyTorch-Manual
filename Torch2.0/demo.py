import torch
from torchvision.models import resnet50

model = resnet50().cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
compiled_model = torch.compile(model)

x = torch.randn(16, 3, 224, 224).cuda()
opt.zero_grad()
out = compiled_model(x)
out.sum().backward()
opt.step()
