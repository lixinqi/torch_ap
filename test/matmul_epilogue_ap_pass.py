import torch
import torch.fx as fx

from tst.torch_ap.ap_pass import ApPass
from tst.torch_ap.match_replace_util import MatchContext
from tst.torch_ap.torch_ap_trace import torch_ap_trace


class PatternModule(torch.nn.Module):
    def forward(self, x):
        return x


class P(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = PatternModule()

    def forward(self, x, y):
        return self.mod(torch.matmul(x, y))


class Replacement(torch.nn.Module):
    def forward(self, x, y):
        # replacement contains 0 call_module nodes
        return torch.matmul(x, y) + 1.0


class MatmulEpilogueApPass(ApPass):
    def pattern(self) -> fx.GraphModule:
        return fx.GraphModule(P(), tracer.trace(P()))

    def constraint(self, match_ctx) -> bool:
        print("[self.get_submodule(match_ctx, 'mod').code]=========")
        print(self.get_submodule(match_ctx, "mod").code)
        print("[self.get_submodule(match_ctx, 'mod').code]=========")
        return True

    def replacement(self, ctx) -> fx.GraphModule:
        return torch_ap_trace(Replacement())


if __name__ == "__main__":

    class MatmulEpilogue(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def forward(self, x):
            return x - self.bias

    class SimpleTracer(fx.Tracer):

        def __init__(self, leaf_module_classes):
            super().__init__()
            self.leaf_module_classes = leaf_module_classes

        def is_leaf_module(self, m, n):
            return isinstance(m, self.leaf_module_classes)

    tracer = SimpleTracer(leaf_module_classes=(PatternModule))

    # Target: Matmul -> MatmulEpilogue (call_module)
    class TargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.epi = MatmulEpilogue(2.0)

        def forward(self, a, b):
            return self.epi(torch.matmul(a, b))

    t_gm = fx.GraphModule(TargetModel(), tracer.trace(TargetModel()))
    from torch.fx.passes.infra.pass_manager import PassManager
    from tst.torch_ap.trivial_ops_folder_pass import TrivialOpsFolderPass

    pass_mgr = PassManager(
        [
            TrivialOpsFolderPass(),
            MatmulEpilogueApPass(),
        ]
    )

    # --- Verification ---
    print("--- Before Transformation ---")
    print(t_gm.code)

    result_gm = pass_mgr(t_gm).graph_module

    print("\n--- After Transformation ---")
    print(result_gm.code)

    # Final assertion: NumOfFilteredOps[replacement, "call_module"] == 0
    final_mods = [n for n in result_gm.graph.nodes if n.op == "call_module"]
    assert len(final_mods) == 0
    print(f"\nRemaining call_module: {len(final_mods)}")
    print("Success: Transformation consistent with input/output protocol.")
