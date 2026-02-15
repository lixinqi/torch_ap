import torch
import torch.fx as fx
import tst.torch_ap.ops as ops


def torch_ap_trace(f):
    if isinstance(f, fx.GraphModule):
        return f
    tracer = fx.Tracer(autowrap_functions=ops.atom_funcs)
    return fx.GraphModule(torch.nn.Module(), tracer.trace(f))
