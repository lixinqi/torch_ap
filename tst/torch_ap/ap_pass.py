import torch
import torch.fx as fx
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace
from tst.torch_ap.match_replace_util import (
    MatchContext,
    fx_graph_match_first_pattern,
    fx_graph_replace_first_pattern,
)


class ApPass:
    def pattern(self) -> fx.GraphModule:
        """$pattern (fx.GraphModule <- ())"""
        raise NotImplementedError

    def constraint(self, match_context) -> bool:
        """$constraint (bool <- MatchContext)"""
        return True

    def replacement(self, match_context) -> fx.GraphModule:
        """$replacement (fx.GraphModule <- MatchContext)"""
        raise NotImplementedError

    def __call__(self, target: fx.GraphModule) -> PassResult:
        """$__call__ (PassResult <- $target fx.GraphModule)"""

        # Execute the transformation logic using the utility function
        gm, modified = fx_graph_replace_first_pattern(
            target, self.pattern, self.constraint, self.replacement
        )

        return PassResult(gm, modified=modified)

    def get_match_context(self, target: fx.GraphModule) -> MatchContext | None:
        return fx_graph_match_first_pattern(target, self.pattern, self.constraint)

    def get_submodule(self, match_ctx, pattern_submodule_name: str) -> fx.GraphModule:
        target_call_module_node = self.get_target_call_module_node(
            match_ctx, pattern_submodule_name
        )
        target_module_name = target_call_module_node.target
        return getattr(match_ctx.target, target_module_name)

    def get_first_target_call_function_node(self, match_ctx, pattern_target):
        pattern_call_function_node = self.get_first_pattern_call_function_node(
            match_ctx, pattern_target
        )
        return match_ctx.nodes_map[pattern_call_function_node]

    def get_first_pattern_call_function_node(self, match_ctx, pattern_target):
        def is_selected_node(node):
            if node.op != "call_function":
                return False
            if node.target != pattern_target:
                return False
            return True

        for node in match_ctx.pattern.graph.nodes:
            if is_selected_node(node):
                return node
        return None

    def get_target_call_module_node(self, match_ctx, pattern_submodule_name: str):
        pattern_call_module_node = self.get_pattern_call_module_node(
            match_ctx, pattern_submodule_name
        )
        return match_ctx.nodes_map[pattern_call_module_node]

    def get_pattern_call_module_node(self, match_ctx, pattern_submodule_name: str):
        def is_selected_call_module_node(node):
            if node.op != "call_module":
                return False
            if node.target != pattern_submodule_name:
                return False
            return True

        for node in match_ctx.pattern.graph.nodes:
            if is_selected_call_module_node(node):
                return node
        return None


if __name__ == "__main__":

    class LinearEpilogue(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def forward(self, x):
            return x + self.bias

    class PatternModule(torch.nn.Module):
        def forward(self, x):
            return x

    class SimpleTracer(fx.Tracer):
        def is_leaf_module(self, m, n):
            return isinstance(m, (LinearEpilogue, PatternModule))

    tracer = SimpleTracer()

    # Target: Matmul -> LinearEpilogue (call_module)
    class TargetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.epi = LinearEpilogue(2.0)

        def forward(self, a, b):
            return self.epi(torch.matmul(a, b))

    t_gm = fx.GraphModule(TargetModel(), tracer.trace(TargetModel()))

    # --- Verification ---
    print("--- Before Transformation ---")
    print(t_gm.code)

    class DemoApPass(ApPass):
        def pattern(self) -> fx.GraphModule:
            class P(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod = PatternModule()

                def forward(self, x, y):
                    return self.mod(torch.matmul(x, y))

            return fx.GraphModule(P(), tracer.trace(P()))

        def replacement(self, ctx) -> fx.GraphModule:
            t_node = next(
                tn for pn, tn in ctx.nodes_map.items() if pn.op == "call_module"
            )
            bias_val = t_gm.get_submodule(t_node.target).bias

            class Replacement(torch.nn.Module):
                def forward(self, x, y):
                    # replacement contains 0 call_module nodes
                    return torch.matmul(x, y) + bias_val

            return torch_ap_trace(Replacement())

    result_gm = DemoApPass()(t_gm).graph_module

    print("\n--- After Transformation ---")
    print(result_gm.code)

    # Final assertion: NumOfFilteredOps[replacement, "call_module"] == 0
    final_mods = [n for n in result_gm.graph.nodes if n.op == "call_module"]
    assert len(final_mods) == 0
    print(f"\nRemaining call_module: {len(final_mods)}")
    print("Success: Transformation consistent with input/output protocol.")
