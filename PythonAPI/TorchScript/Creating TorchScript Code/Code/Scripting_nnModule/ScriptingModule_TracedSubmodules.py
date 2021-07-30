# scripting a module with traced submodules
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        # torch.jit.trace produces the ScriptModule's conv1 and conv2
        self.conv1 = torch.jit.trace(nn.Conv2d(1, 10, 5), torch.rand(1, 1, 16, 16))
        self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

scripted_module = torch.jit.script(MyModule()) 
print(scripted_module)