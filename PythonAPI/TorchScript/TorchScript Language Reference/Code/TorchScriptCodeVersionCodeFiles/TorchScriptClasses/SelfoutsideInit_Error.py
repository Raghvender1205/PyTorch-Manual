'''
RuntimeError: 
Tried to set nonexistent attribute: x. Did you forget to initialize it in __init__()?:
  File "D:\ML\PyTorch\PyTorch_Manual\TorchScript Language Reference\Code\TorchScriptCodeVersionCodeFiles\TorchScriptClasses\SelfoutsideInit_Error.py", line 6
    def assign_x(self):
        self.x = torch.rand(2, 3)
        ~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
'''

import torch

@torch.jit.script
class Foo:
    def assign_x(self):
        self.x = torch.rand(2, 3)
