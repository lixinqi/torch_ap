import torch
import torch.fx as fx
from typing import List


class OutputAsMutInputsTransformer:
    def __init__(self):
        # __init__ <- ()
        pass

    def __call__(
        self,
        target: fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> fx.GraphModule:
        # 1. ($graph_module_with_sole_submodule <- $target)
        gm_with_sub, placeholder_nodes = self._fold_to_sole_submodule(target)

        # Extract dtypes and shapes from example_inputs
        if isinstance(example_inputs, (tuple, list)):
            input_dtypes = [inp.dtype for inp in example_inputs]
            example_input_shapes = [list(inp.shape) for inp in example_inputs]
        else:
            input_dtypes = [example_inputs.dtype]
            example_input_shapes = [list(example_inputs.shape)]

        # 2. ($output_shapes <- $target <- $example_inputs)
        output_shapes = self._infer_output_shapes(
            target, example_inputs
        )

        # 3. ($inserted_mut_input_nodes <- $gm_with_sub <- $example_input_shapes <- $output_shapes <- $placeholder_nodes)
        mut_input_nodes = self._insert_empty_nodes(
            gm_with_sub, output_shapes, input_dtypes, placeholder_nodes
        )

        # 4. ($sole_submodule <- $gm_with_sub)
        sole_sub = next(
            m
            for m in gm_with_sub.modules()
            if isinstance(m, fx.GraphModule) and m is not gm_with_sub
        )

        # 5. ($sole_submodule_with_mut_inputs <- $sole_sub <- $inserted_mut_input_nodes)
        self._add_mut_placeholders_to_submodule(sole_sub, mut_input_nodes)

        # 6. ($sole_submodule_without_outputs <- $sole_submodule_with_mut_inputs)
        self._convert_outputs_to_mut_ops(sole_sub)

        # 7. Set the output to return mut_input nodes (the tensors that were modified in-place)
        gm_with_sub.graph.output(mut_input_nodes[0] if len(mut_input_nodes) == 1 else tuple(mut_input_nodes))
        gm_with_sub.recompile()

        return gm_with_sub

    def _fold_to_sole_submodule(self, target: fx.GraphModule) -> fx.GraphModule:
        """Inline logic: Folds existing graph into a submodule called 'sub'."""
        # First create an empty GM, then add submodule to avoid empty graph issue during recompile
        new_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        new_gm.add_submodule("sub", target)

        # Now build the graph
        new_graph = new_gm.graph
        placeholder_nodes = []
        for n in target.graph.nodes:
            if n.op == "placeholder":
                new_ph = new_graph.placeholder(n.target)
                placeholder_nodes.append(new_ph)

        # Call the submodule
        sub_call = new_graph.call_module("sub", args=tuple(placeholder_nodes))
        # Note: Output is set later in __call__ after mut_input nodes are inserted
        new_gm.recompile()
        return new_gm, placeholder_nodes

    def _infer_output_shapes(
        self,
        target: fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> List[List[torch.SymInt]]:
        """Infer output shapes by running the model with example_inputs in FakeTensorMode."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # Use FakeTensorMode to capture symbolic shapes (sym.Int)
        with FakeTensorMode(shape_env=ShapeEnv()) as mode:
            # Convert example_inputs to FakeTensors to enable symbolic reasoning
            fake_inputs = []
            for t in example_inputs:
                if isinstance(t, torch._subclasses.fake_tensor.FakeTensor):
                    # If it's already a FakeTensor, ensure it's in OUR mode to avoid "Mixing fake modes"
                    fake_inputs.append(mode.from_tensor(t))
                else:
                    fake_inputs.append(mode.from_tensor(t))
            
            with torch.no_grad():
                outputs = target(*fake_inputs)
            
            if isinstance(outputs, (tuple, list)):
                return [list(o.shape) for o in outputs]
            return [list(outputs.shape)]

    def _insert_empty_nodes(self, gm, out_shapes, dtypes, placeholder_nodes) -> List[fx.Node]:
        """Inline logic: Insert torch.empty at the beginning of the main graph.
        Supports symbolic shapes (list[list[sym.Int]]).
        """
        first_node = next(iter(gm.graph.nodes))
        inserted = []
        with gm.graph.inserting_before(first_node):
            for shape in out_shapes:
                node = gm.graph.call_function(
                    torch.empty, args=(tuple(shape),), kwargs={"dtype": dtypes[0]}
                )
                inserted.append(node)

        # Update the call to 'sub' with the newly inserted empty tensors
        # Order: (placeholder_nodes..., mut_input_nodes...)
        sub_node = next(
            n for n in gm.graph.nodes if n.op == "call_module" and n.target == "sub"
        )
        sub_node.args = (*placeholder_nodes, *inserted)
        gm.recompile()
        return inserted

    def _add_mut_placeholders_to_submodule(
        self, sub_gm: fx.GraphModule, mut_nodes: List[fx.Node]
    ):
        """Inline logic: Add placeholders to the submodule's graph."""
        with sub_gm.graph.inserting_after(None):  # Insert at beginning
            # Find the last placeholder and insert after it
            last_ph = None
            for n in sub_gm.graph.nodes:
                if n.op == "placeholder":
                    last_ph = n

            with sub_gm.graph.inserting_after(last_ph):
                for i in range(len(mut_nodes)):
                    sub_gm.graph.placeholder(f"mut_input_{i}")
        sub_gm.recompile()

    def _convert_outputs_to_mut_ops(self, sub_gm: fx.GraphModule):
        """Inline logic: Replace return with in-place copy (copy_)."""
        output_node = next(n for n in sub_gm.graph.nodes if n.op == "output")
        mut_placeholders = [
            n
            for n in sub_gm.graph.nodes
            if n.op == "placeholder" and "mut_input_" in n.target
        ]

        # Assume output is a Tensor or Tuple[Tensor]
        out_vals = output_node.args[0]
        if not isinstance(out_vals, (tuple, list)):
            out_vals = [out_vals]

        with sub_gm.graph.inserting_before(output_node):
            for val, ph in zip(out_vals, mut_placeholders):
                # Core transformation: use copy_ for in-place assignment
                sub_gm.graph.call_method("copy_", args=(ph, val))

        # Remove return value (return void)
        output_node.args = (None,)
        sub_gm.recompile()


def test_main():
    # Construct a simple computation graph: out = x + y
    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    transformer = OutputAsMutInputsTransformer()
    # Provide example_inputs to infer output shapes, dtypes, and input shapes
    example_inputs = (torch.randn(128, 64), torch.randn(128, 64))
    new_gm = transformer(gm, example_inputs)

    print("--- Transformed Graph Module Code ---")
    print(new_gm.code)
    print("\n--- Sole Submodule Code (Side Effect Version) ---")
    print(new_gm.sub.code)


if __name__ == "__main__":
    test_main()
