import torch
import torch.fx as fx
from torch.fx.passes.infra.pass_manager import PassResult
from typing import Any
from tst.torch_ap.torch_ap_trace import torch_ap_trace


class PatternReplacerPass:
    """
    PatternReplacerPass := PassResult <- $target <- ($p_func, $r_func)
    """

    def __init__(self, pattern_func: Any, replacement_func: Any):
        # # __init__ currying
        self.pattern_func = pattern_func
        self.replacement_func = replacement_func

    def __call__(self, target: fx.GraphModule) -> PassResult:
        # # inline: fx.subgraph_rewriter.replace_pattern
        matches = fx.subgraph_rewriter.replace_pattern(
            target,
            torch_ap_trace(self.pattern_func),
            torch_ap_trace(self.replacement_func),
        )

        return PassResult(graph_module=target, modified=len(matches) > 0)


# --- Test Environment ---


def main():
    # 1. Setup target module: y = x + x
    class SimpleModule(torch.nn.Module):
        def forward(self, x):
            val = x + x
            return torch.relu(val)

    # 2. Define Viba Builders
    def pattern(x):
        return x + x

    def replacement(x):
        return x * 2

    target_gm = torch_ap_trace(SimpleModule())

    # --- PRINT BEFORE ---
    print("=== Graph BEFORE Transformation ===")
    target_gm.graph.print_tabular()
    print("-" * 40)

    # 3. Initialize and Execute Pass
    replacer = PatternReplacerPass(pattern, replacement)
    result = replacer(target_gm)

    # --- PRINT AFTER ---
    print(f"=== Graph AFTER Transformation (Modified: {result.modified}) ===")
    if result.modified:
        result.graph_module.graph.print_tabular()
    else:
        print("No matches found. Graph remains unchanged.")


if __name__ == "__main__":
    main()
