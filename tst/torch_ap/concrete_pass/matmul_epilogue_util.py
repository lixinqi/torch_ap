import torch
import torch.fx as fx

from tst.torch_ap.ap_pass import ApPass, PassResult
from tst.torch_ap.match_replace_util import MatchContext


def get_matmul_epilogue_arg_name_to_is_mm_out(graph_module) -> list[(str, bool)]:
    return MatmulEpilogueArgNameToIsMmOutGetter()(graph_module)


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


class MatmulEpilogueArgNameToIsMmOutGetter(ApPass):
    def pattern(self) -> fx.GraphModule:
        tracer = SimpleTracer(leaf_module_classes=(PatternModule))
        return fx.GraphModule(P(), tracer.trace(P()))

    def constraint(self, match_ctx) -> bool:
        return True

    def __call__(self, target: fx.GraphModule) -> list[tuple[str, bool]]:
        match_ctx = self.get_match_context(target)
        if match_ctx is None:
            return []
        placeholder_names = self._get_epilogue_placeholder_names(match_ctx)
        arg_is_mm_out_list = self._get_arg_is_mm_out_list(match_ctx)
        return list(zip(placeholder_names, arg_is_mm_out_list, strict=True))

    def _get_arg_is_mm_out_list(self, match_ctx) -> list[bool]:
        matmul_node = self.get_first_target_call_function_node(match_ctx, torch.matmul)
        call_module = self.get_target_call_module_node(match_ctx, "mod")
        return [arg is matmul_node for arg in call_module.args]

    def _get_epilogue_placeholder_names(self, match_ctx) -> list[str]:
        epilogue_module = self.get_submodule(match_ctx, pattern_submodule_name="mod")
        return [
            node.name
            for node in epilogue_module.graph.nodes
            if node.op == "placeholder"
        ]


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

    pass_mgr = TrivialOpsFolderPass()

    # --- Verification ---
    print("--- Before Transformation ---")
    print(t_gm.code)

    result = MatmulEpilogueArgNameToIsMmOutGetter()(pass_mgr(t_gm).graph_module)

    print("\n--- After Transformation ---")
    print(result)
