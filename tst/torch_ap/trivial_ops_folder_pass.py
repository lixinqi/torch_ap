import torch.fx as fx
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace

# --- ImportFrom (Viba defined mappings) ---
from tst.torch_ap.trivial_ops_util import get_trivial_ops_ranges, is_trivial_op
from tst.torch_ap.submodule_fold_util import convert_to_submodules_graph

# --- Core Functor Implementation ---


class TrivialOpsFolderPass:
    """
    TrivialOpsFolderPass := PassResult <- fx.GraphModule <- $is_trivial_op
    """

    def __init__(self, is_trivial_op_fn=is_trivial_op):
        # Captured via __init__ (currying splitter)
        self.is_trivial_op = is_trivial_op_fn

    def __call__(self, gm: fx.GraphModule) -> PassResult:
        # 1. Identify ranges using absolute indices ($ranges_start_from_placeholder)
        raw_ranges = get_trivial_ops_ranges(gm, self.is_trivial_op)

        if not raw_ranges:
            return PassResult(graph_module=gm, modified=False)

        # 2. Dynamic Offset Calculation
        # Maps absolute graph indices to non-placeholder indices
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        num_placeholders = len(placeholders)

        # Adjust to $ranges_start_from_non_placeholder
        adjusted_ranges = [
            (s - num_placeholders, e - num_placeholders) for s, e in raw_ranges
        ]

        # 3. Fold ranges into submodules
        folded_gm = convert_to_submodules_graph(gm, adjusted_ranges)

        return PassResult(graph_module=folded_gm, modified=True)


# --- Test Environment ---


def main():
    import torch
    import operator

    # Scenario: Multi-input model
    class M(torch.nn.Module):
        def forward(self, x, y):
            z = x + y  # Trivial (operator.add)
            z = torch.relu(z)  # Trivial (torch.relu)
            z = z * 2  # Trivial (operator.mul)
            return z

    gm = torch_ap_trace(M())

    print("--- [Before Transformation] ---")
    gm.graph.print_tabular()

    # Use the imported is_trivial_op which now includes add/relu/mul
    folder = TrivialOpsFolderPass(is_trivial_op)
    result = folder(gm)

    if result.modified:
        print("\n--- [After Transformation] ---")
        result.graph_module.graph.print_tabular()

        # Inspecting the result: submodule should contain the 3 ops
        for name, module in result.graph_module.named_modules():
            if "submodule" in name:
                print(f"\n--- [Folded Submodule: {name}] ---")
                print(module.code)


if __name__ == "__main__":
    main()
