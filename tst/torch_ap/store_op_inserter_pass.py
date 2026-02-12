import torch
import torch.fx as fx
from typing import List, Union
from tst.torch_ap.load_store_op import store
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace


class StoreOpInserterPass:
    def __init__(self):
        """
        # __init__ <- ()
        """
        pass

    def __call__(self, target: fx.GraphModule) -> PassResult:
        """
        # __call__ PassResult <- $target fx.GraphModule
        """
        graph = target.graph
        modified = False

        # Locate the unique output node
        output_node = next((n for n in graph.nodes if n.op == "output"), None)
        if output_node is None:
            return PassResult(graph_module=target, modified=False)

        def create_store_node(val: fx.Node, idx: Union[int, None]) -> fx.Node:
            """
            Internal helper to create the store call.
            """
            return graph.call_function(store, args=(val, idx))

        def transform_collection(args_collection: Union[tuple, list]):
            """
            # $insert_store_op_before_tuple_output (void <- fx.Node <- $output_idx int)
            Handles the iteration to keep the main logic shallow.
            """
            return [
                create_store_node(arg, i) if isinstance(arg, fx.Node) else arg
                for i, arg in enumerate(args_collection)
            ]

        def handle_insertion():
            nonlocal modified
            # # inline: $get_output_node_args (list[fx.Node] <- $target)
            output_args = output_node.args[0]

            with graph.inserting_before(output_node):
                if isinstance(output_args, fx.Node):
                    # # $insert_store_op_before_sole_output (void <- fx.Node <- $output_idx None)
                    new_node = create_store_node(output_args, None)
                    output_node.args = (new_node,)
                    modified = True
                elif isinstance(output_args, (tuple, list)):
                    # # $insert_store_op_before_tuple_output (void <- fx.Node <- $output_idx int)
                    new_list = transform_collection(output_args)
                    output_node.args = (
                        tuple(new_list) if isinstance(output_args, tuple) else new_list,
                    )
                    modified = True

        handle_insertion()

        if modified:
            target.recompile()

        return PassResult(graph_module=target, modified=modified)


def test_main():
    """
    Test cases for both single output and multi-output (tuple) scenarios.
    """

    # Case 1: Single Output - store(x, None)
    class SingleOutputModel(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    print("Testing Case 1: Single Output...")
    traced_single = torch_ap_trace(SingleOutputModel())
    res_single = StoreOpInserterPass()(traced_single)
    res_single.graph_module.graph.print_tabular()

    # Verification
    input_data = torch.randn(2)
    assert torch.allclose(
        SingleOutputModel()(input_data), res_single.graph_module(input_data)
    )
    print("Single output test passed.\n")

    # Case 2: Multi-Output (Tuple) - store(x, int)
    class MultiOutputModel(torch.nn.Module):
        def forward(self, x, y):
            a = x * 2.0
            b = y / 2.0
            return a, b

    print("Testing Case 2: Multi-Output...")
    traced_multi = torch_ap_trace(MultiOutputModel())
    res_multi = StoreOpInserterPass()(traced_multi)
    res_multi.graph_module.graph.print_tabular()

    # Verification
    x_val, y_val = torch.randn(2), torch.randn(2)
    ref_a, ref_b = MultiOutputModel()(x_val, y_val)
    test_a, test_b = res_multi.graph_module(x_val, y_val)
    assert torch.allclose(ref_a, test_a) and torch.allclose(ref_b, test_b)
    print("Multi-output test passed.")


if __name__ == "__main__":
    test_main()
