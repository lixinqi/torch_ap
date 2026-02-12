import torch
import torch.fx as fx

from tst.torch_ap.ap_pass import ApPass, PassResult
from tst.torch_ap.match_replace_util import MatchContext


class SimpleTracer(fx.Tracer):

    def __init__(self, leaf_module_classes):
        super().__init__()
        self.leaf_module_classes = leaf_module_classes

    def is_leaf_module(self, m, n):
        return isinstance(m, self.leaf_module_classes)


class PatternModule(torch.nn.Module):
    def forward(self, x):
        return x


class P(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = PatternModule()

    def forward(self, x, y):
        return self.mod(torch.matmul(x, y))


class MatmulEpilogueExtractorPass(ApPass):
    def pattern(self) -> fx.GraphModule:
        tracer = SimpleTracer(leaf_module_classes=(PatternModule))
        return fx.GraphModule(P(), tracer.trace(P()))

    def constraint(self, match_ctx) -> bool:
        return True

    def __call__(self, target: fx.GraphModule) -> PassResult:
        match_ctx = self.get_match_context(target)
        if match_ctx is None:
            return PassResult(target, modified=False)
        epilogue_module = self.get_submodule(match_ctx, pattern_submodule_name="mod")
        return PassResult(epilogue_module, modified=True)


if __name__ == "__main__":

    # Target: Matmul -> MatmulEpilogue (call_module)
    class TargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.tanh(torch.matmul(a, b) - 2.0)

    t_gm = fx.GraphModule(TargetModel(), fx.Tracer().trace(TargetModel()))
    from torch.fx.passes.infra.pass_manager import PassManager
    from tst.torch_ap.trivial_ops_folder_pass import TrivialOpsFolderPass

    pass_mgr = PassManager(
        [
            TrivialOpsFolderPass(),
            MatmulEpilogueExtractorPass(),
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
