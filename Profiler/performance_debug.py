import numpy as np
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

class Sample(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Sample, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, mask):
        with profiler.record_function('LINEAR PASS'):
            out = self.linear(x)
        
        with profiler.record_function('MASK INDICES'):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx
    
"""
Copying a Matrix from CUDA to CPU is expensive. aten::copy_ operator copies mask to CPU so that
it could use Numpy's argwhere function and then aten::copy_ copies the array back to CUDA as a Tensor.

It can be removed by using torch.nonzero() function
"""
class ImprovedSample(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ImprovedSample, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, mask):
        with profiler.record_function('LINEAR PASS'):
            out = self.linear(x)
        
        with profiler.record_function('MASK INDICES'):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx

if __name__ == '__main__':
    model = ImprovedSample(500, 100).cuda()
    x = torch.rand(128, 500).cuda()
    mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

    # Run and Profile
    model(x, mask)
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        out, idx = model(x, mask)
    
    # Print results
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))