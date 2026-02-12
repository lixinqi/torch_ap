import torch
import torch.fx as fx
from torch.fx import subgraph_rewriter
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace


class PatternRemoverPass:
    def __init__(self, pattern_func):
        self.pattern_func = pattern_func

        # Trace pattern_func directly
        self.pattern_gm = torch_ap_trace(self.pattern_func)

        # 1. $get_num_placehoders
        def get_num_placeholders(gm: fx.GraphModule) -> int:
            return len([n for n in gm.graph.nodes if n.op == "placeholder"])

        # 2. $get_num_output_node_args
        def get_num_output_node_args(gm: fx.GraphModule) -> int:
            output_node = next(n for n in gm.graph.nodes if n.op == "output")
            out_args = output_node.args[0]
            if out_args is None:
                return 0
            return len(out_args) if isinstance(out_args, (tuple, list)) else 1

        # 3. $truncate_placeholders_as_new_outputs
        def truncate_placeholders_as_new_outputs(placeholders, num_out):
            return placeholders[:num_out]

        # 4. $extend_outputs_with_none_if_num_inputs_less_than_outputs
        def extend_outputs(nodes, final_size):
            return nodes + [None] * (final_size - len(nodes))

        # Synthesize Replacement Graph
        replace_graph = fx.Graph()
        num_inputs = get_num_placeholders(self.pattern_gm)
        num_outputs = get_num_output_node_args(self.pattern_gm)

        new_placeholders = [
            replace_graph.placeholder(f"arg_{i}") for i in range(num_inputs)
        ]

        identity_nodes = truncate_placeholders_as_new_outputs(
            new_placeholders, num_outputs
        )
        final_outputs = extend_outputs(identity_nodes, num_outputs)

        # Handle single vs multiple output return format
        output_val = (
            tuple(final_outputs)
            if len(final_outputs) > 1
            else (final_outputs[0] if final_outputs else None)
        )
        replace_graph.output(output_val)

        self.replacement_gm = fx.GraphModule(torch.nn.Module(), replace_graph)

    def __call__(self, gm: fx.GraphModule) -> PassResult:
        # Match pattern and bypass using identity replacement
        matches = subgraph_rewriter.replace_pattern(
            gm, torch_ap_trace(self.pattern_gm), torch_ap_trace(self.replacement_gm)
        )

        modified = len(matches) > 0
        if modified:
            gm.recompile()

        return PassResult(graph_module=gm, modified=modified)


# --- Test Environment ---


def main():
    import operator

    # Pattern: pow(x, 2)
    def power_pattern(x):
        return operator.pow(x, 2)

    class M(torch.nn.Module):
        def forward(self, x):
            y = x**2
            return y + 5

    gm = torch_ap_trace(M())

    print("--- [Before] ---")
    gm.graph.print_tabular()

    remover = PatternRemoverPass(power_pattern)
    result = remover(gm)

    if result.modified:
        print("\n--- [After Removal (Bypassed to Input)] ---")
        result.graph_module.graph.print_tabular()


if __name__ == "__main__":
    main()
