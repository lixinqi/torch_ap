import torch
import torch.fx as fx
from typing import List
from tst.torch_ap.load_store_op import load
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace


class LoadOpInserterPass:
    def __init__(self):
        """
        Initialize the pass with no arguments.
        """
        pass

    def __call__(self, target: fx.GraphModule) -> PassResult:
        """
        Transform the graph by inserting explicit load operations after each placeholder.
        """
        graph = target.graph
        modified = False

        # Identify all placeholder nodes representing input arguments
        placeholders: List[fx.Node] = [n for n in graph.nodes if n.op == "placeholder"]

        for ph in placeholders:
            # Create a load node immediately after the placeholder
            # Use the placeholder's target name as the semantic buffer name
            with graph.inserting_after(ph):
                load_node = graph.call_function(load, args=(ph, ph.target))

                # Replace all downstream uses of the placeholder with the load node
                # Filter out the load node itself to maintain DAG integrity
                ph.replace_all_uses_with(
                    load_node, delete_user_cb=lambda user: user != load_node
                )
                modified = True

        if modified:
            target.recompile()

        return PassResult(graph_module=target, modified=modified)


def test_main():
    """
    Test the LoadOpInserterPass on a simple compute graph.
    """

    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    # Trace the model
    model = SimpleModel()
    traced = torch_ap_trace(model)

    # print("Graph before LoadOpInserterPass:")
    # traced.graph.print_tabular()

    # Apply the pass
    inserter = LoadOpInserterPass()
    result = inserter(traced)

    print(f"\nModified: {result.modified}")
    # print("Graph after LoadOpInserterPass:")
    # result.graph_module.graph.print_tabular()

    # Functional verification
    x, y = torch.randn(3), torch.randn(3)
    original_out = model(x, y)
    transformed_out = result.graph_module(x, y)

    assert torch.equal(original_out, transformed_out), "Functionality broken!"
    print("\nTest passed: Computational integrity maintained.")


if __name__ == "__main__":
    test_main()
