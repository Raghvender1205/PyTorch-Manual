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

#print(prof)
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

class ProfilerTracer(torch.fx.Tracer):
    def trace(self, root, concrete_args=None):
        # Original Autograd Profile Record Functions...!!
        orig_record_function_enter = torch.autograd.profiler.record_function.__enter__
        orig_record_function_exit = torch.autograd.profiler.record_function.__exit__

        # Fake Profile Record Enter and Exit Functions
        def fake_profiler_enter(_self):
            nonlocal self
            handle_proxy = self.create_proxy(
                kind='call_function',
                target=torch.ops.profiler._record_function_enter,
                args=(_self.name,),
                kwargs={})
        
            assert getattr(_self, '_fx_profiler_ctx', None) is None
            setattr(_self, '_fx_profiler_ctx', handle_proxy)
            return handle_proxy
        
        def fake_profiler_exit(_self, exc_type, exc_value, traceback):
            assert hasattr(_self, '_fx_profiler_ctx')
            handle_proxy = _self._fx_profiler_ctx
            torch.ops.profiler._record_function_exit(handle_proxy)
            setattr(_self, '_fx_profiler_ctx', None)

        torch.autograd.profiler.record_function.__enter__ = fake_profiler_enter
        torch.autograd.profiler.record_function.__exit__ = fake_profiler_exit

        try:
            return super().trace(root, concrete_args)
        finally:
            torch.autograd.profiler.record_function.__enter__ = orig_record_function_enter
            torch.autograd.profiler.record_function.__exit__ = orig_record_function_exit


pt = ProfilerTracer()

graph_with_profiler = pt.trace(f)
traced_with_profiler = torch.fx.GraphModule(pt.root, graph_with_profiler)

with torch.autograd.profiler.profile() as prof:
    traced_with_profiler(x)

print(prof)
# "foo" range is recorded with special tracer behaviour
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
                foo        51.35%      19.000us       100.00%      37.000us      37.000us             1
        aten::empty        10.81%       4.000us        10.81%       4.000us       4.000us             1
         aten::relu         8.11%       3.000us        37.84%      14.000us      14.000us             1
    aten::clamp_min        10.81%       4.000us        29.73%      11.000us      11.000us             1
        aten::empty         2.70%       1.000us         2.70%       1.000us       1.000us             1
    aten::clamp_min        16.22%       6.000us        16.22%       6.000us       6.000us             1
-------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 37.000us
"""
