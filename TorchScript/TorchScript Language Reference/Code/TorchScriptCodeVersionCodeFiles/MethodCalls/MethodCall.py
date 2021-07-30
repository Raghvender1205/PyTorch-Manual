import torch
import torch.nn as nn
import torchvision

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        means = torch.tensor([103.939, 116.779, 123.68])
        self.means = nn.Parameter(means.resize_(1, 3, 1, 1))
        resnet = torchvision.models.resnet18()
        self.resnet = torch.jit.trace(resnet, torch.rand(1, 3, 224, 224))
    
    def helper(self, input):
        return self.resnet(input - self.means)
    
    def forward(self, input):
        return self.helper(input)
    
    # Since nothing in the model calls `top_level_method`, the compiler
    # must be explicitly told to compile this method
    @torch.jit.export
    def top_level_method(self, input):
        return self.other_helper(input)
    
    def other_helper(self, input):
        return input + 10


# `my_script_module` will have the compiled methods `forward`, `helper` and the others
my_script_module = torch.jit.script(MyModule())