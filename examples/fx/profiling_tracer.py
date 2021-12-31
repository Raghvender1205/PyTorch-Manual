"""
We demonstrate here using a custom FX Tracer to override the behavior of
`torch.autograd.profiler.record_function` and make profiler appear in
FX-Traced Code.

This is done with Python dynamic patching magic, allowing us to explicitly emit calls to
`torch.ops.profiler._record_function_enter/_record_function_exit`.

Although, these ranges may be eliminated by `Graph.eliminate_dead_code`.
"""
import torch
import torch.fx

# NOTE: A module with `record_function`
class Foo(torch.nn.Module):
    def forward(self, x):
        with torch.profiler.record_function('foo'):
            return torch.relu(x)

f = Foo()
x = torch.randn(5, 3, 2)
with torch.autograd.profiler.profile() as prof:
    f(x)

#print(prof)
# "foo" range is correctly recorded with normal execution
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
        aten::zeros        36.45%      14.149ms        38.72%      15.030ms      15.030ms             1
        aten::empty         0.85%     329.000us         0.85%     329.000us     329.000us             1
        aten::zero_         1.42%     552.000us         1.42%     552.000us     552.000us             1
                foo        28.53%      11.075ms        61.28%      23.787ms      23.787ms             1
        aten::empty         0.03%      11.000us         0.03%      11.000us      11.000us             1
         aten::relu        10.90%       4.231ms        32.72%      12.701ms      12.701ms             1
    aten::clamp_min         4.20%       1.632ms        21.82%       8.470ms       8.470ms             1
        aten::empty         0.02%       9.000us         0.02%       9.000us       9.000us             1
    aten::clamp_min        17.59%       6.829ms        17.59%       6.829ms       6.829ms             1
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 38.817ms
"""

traced = torch.fx.symbolic_trace(f)
with torch.autograd.profiler.profile() as prof:
    traced(x)

print(prof)
# "foo" range is not recorded with FX Tracing
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
         aten::relu        18.18%       4.000us       100.00%      22.000us      22.000us             1
    aten::clamp_min        31.82%       7.000us        81.82%      18.000us      18.000us             1
        aten::empty        18.18%       4.000us        18.18%       4.000us       4.000us             1
    aten::clamp_min        31.82%       7.000us        31.82%       7.000us       7.000us             1
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 22.000us
"""
