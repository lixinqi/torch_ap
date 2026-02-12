import torch
import torch.fx as fx
from typing import List, Dict, Any, Tuple, Union
import operator


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

        # Extract dtypes from example_inputs
        if isinstance(example_inputs, (tuple, list)):
            input_dtypes = [inp.dtype for inp in example_inputs]
        else:
            input_dtypes = [example_inputs.dtype]

        # 2. ($symbolic_output_shapes * $input_symbol_map <- $target <- $input_dtypes <- $symbolic_input_shapes <- $placeholder_nodes)
        # We need to capture the symbolic environment to trace symbol sources
        symbolic_output_shapes, input_symbol_map = self._infer_output_shapes_with_sources(
            target, example_inputs, placeholder_nodes
        )

        # 3. ($runnable_output_shapes <- $symbolic_output_shapes <- $input_symbol_map)
        runnable_output_shapes = self._bind_symbols_to_nodes(
            symbolic_output_shapes, input_symbol_map
        )

        # 4. ($inserted_mut_input_nodes <- $gm_with_sub <- $runnable_output_shapes)
        mut_input_nodes = self._insert_empty_nodes(
            gm_with_sub, runnable_output_shapes, input_dtypes, placeholder_nodes
        )

        # 5. ($sole_submodule <- $gm_with_sub)
        sole_sub = next(
            m
            for m in gm_with_sub.modules()
            if isinstance(m, fx.GraphModule) and m is not gm_with_sub
        )

        # 6. ($sole_submodule_with_mut_inputs <- $sole_sub <- $inserted_mut_input_nodes)
        self._add_mut_placeholders_to_submodule(sole_sub, mut_input_nodes)

        # 7. ($sole_submodule_without_outputs <- $sole_submodule_with_mut_inputs)
        self._convert_outputs_to_mut_ops(sole_sub)

        # Finalize the main graph output
        gm_with_sub.graph.output(mut_input_nodes[0] if len(mut_input_nodes) == 1 else tuple(mut_input_nodes))
        gm_with_sub.recompile()

        return gm_with_sub

    def _fold_to_sole_submodule(self, target: fx.GraphModule) -> fx.GraphModule:
        """Inline logic: Folds existing graph into a submodule called 'sub'."""
        new_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        new_gm.add_submodule("sub", target)

        # Now build the graph
        new_graph = new_gm.graph
        placeholder_nodes = []
        for n in target.graph.nodes:
            if n.op == "placeholder":
                new_ph = new_graph.placeholder(n.target)
                placeholder_nodes.append(new_ph)

        new_graph.call_module("sub", args=tuple(placeholder_nodes))
        new_gm.recompile()
        return new_gm, placeholder_nodes

    def _infer_output_shapes_with_sources(
        self,
        target: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        placeholders: List[fx.Node]
    ) -> Tuple[List[List[torch.SymInt]], Dict[Any, Tuple[fx.Node, int]]]:
        """
        Infer output shapes and build a map from SymInt to (placeholder_node, dim_index).
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        input_symbol_map = {}
        with FakeTensorMode(shape_env=ShapeEnv()) as mode:
            fake_inputs = [mode.from_tensor(t) for t in example_inputs]
            
            # Build the map: SymInt -> (FX Placeholder Node, Dimension Index)
            for fake_input, ph_node in zip(fake_inputs, placeholders):
                for i, dim in enumerate(fake_input.shape):
                    if isinstance(dim, torch.SymInt):
                        # Use the underlying node/expr as a hashable key
                        # In newer PyTorch, SymInt has a .node attribute
                        key = dim.node if hasattr(dim, "node") else str(dim)
                        input_symbol_map[key] = (ph_node, i)
            
            with torch.no_grad():
                outputs = target(*fake_inputs)
            
            if isinstance(outputs, (tuple, list)):
                symbolic_shapes = [list(o.shape) for o in outputs]
            else:
                symbolic_shapes = [list(outputs.shape)]
                
        return symbolic_shapes, input_symbol_map

    def _bind_symbols_to_nodes(
        self, 
        symbolic_shapes: List[List[torch.SymInt]], 
        input_symbol_map: Dict[Any, Tuple[fx.Node, int]]
    ) -> List[List[Union[int, Tuple[fx.Node, int]]]]:
        """
        3. ($runnable_output_shapes <- $symbolic_output_shapes <- $placeholder_nodes)
        Now uses the input_symbol_map to find the EXACT source for each SymInt.
        """
        runnable_shapes = []
        for shape in symbolic_shapes:
            runnable_dim_recipe = []
            for dim in shape:
                if isinstance(dim, int):
                    runnable_dim_recipe.append(dim)
                elif isinstance(dim, torch.SymInt):
                    key = dim.node if hasattr(dim, "node") else str(dim)
                    if key in input_symbol_map:
                        runnable_dim_recipe.append(input_symbol_map[key])
                    else:
                        runnable_dim_recipe.append(dim)
                else:
                    runnable_dim_recipe.append(dim)
            runnable_shapes.append(runnable_dim_recipe)
        return runnable_shapes

    def _insert_empty_nodes(
        self, 
        gm: fx.GraphModule, 
        runnable_shapes: List[List[Union[int, Tuple[fx.Node, int]]]], 
        dtypes: List[torch.dtype],
        placeholders: List[fx.Node]
    ) -> List[fx.Node]:
        """
        4. ($inserted_mut_input_nodes <- $gm_with_sub <- $runnable_output_shapes)
        Materializes the recipe into the FX graph with caching to avoid redundant getattr calls.
        """
        insert_point = None
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                insert_point = node
                break
        
        inserted_empty_nodes = []
        shape_node_cache = {} # Cache for getattr(node, 'shape')
        
        with gm.graph.inserting_before(insert_point):
            for recipe in runnable_shapes:
                actual_shape_nodes = []
                for item in recipe:
                    if isinstance(item, int):
                        actual_shape_nodes.append(item)
                    elif isinstance(item, tuple):
                        source_node, dim_idx = item
                        # Use cache to avoid redundant getattr(x, 'shape')
                        if source_node not in shape_node_cache:
                            shape_node_cache[source_node] = gm.graph.call_function(
                                getattr, args=(source_node, "shape")
                            )
                        shape_attr = shape_node_cache[source_node]
                        dim_val = gm.graph.call_function(operator.getitem, args=(shape_attr, dim_idx))
                        actual_shape_nodes.append(dim_val)
                    else:
                        # Handle cases where SymInt couldn't be bound (should not happen in this demo)
                        actual_shape_nodes.append(item)
                
                empty_node = gm.graph.call_function(
                    torch.empty, 
                    args=(tuple(actual_shape_nodes),), 
                    kwargs={"dtype": dtypes[0]}
                )
                inserted_empty_nodes.append(empty_node)

        sub_node = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "sub")
        sub_node.args = (*placeholders, *inserted_empty_nodes)
        gm.recompile()
        return inserted_empty_nodes

    def _add_mut_placeholders_to_submodule(
        self, sub_gm: fx.GraphModule, mut_nodes: List[fx.Node]
    ):
        """6. ($sole_submodule_with_mut_inputs <- $sole_sub <- $inserted_mut_input_nodes)"""
        last_ph = None
        for n in sub_gm.graph.nodes:
            if n.op == "placeholder":
                last_ph = n

        with sub_gm.graph.inserting_after(last_ph):
            for i in range(len(mut_nodes)):
                sub_gm.graph.placeholder(f"mut_input_{i}")
        sub_gm.recompile()

    def _convert_outputs_to_mut_ops(self, sub_gm: fx.GraphModule):
        """7. ($sole_submodule_without_outputs <- $sole_submodule_with_mut_inputs)"""
        output_node = next(n for n in sub_gm.graph.nodes if n.op == "output")
        mut_placeholders = [
            n
            for n in sub_gm.graph.nodes
            if n.op == "placeholder" and "mut_input_" in n.target
        ]

        out_vals = output_node.args[0]
        if not isinstance(out_vals, (tuple, list)):
            out_vals = [out_vals]

        with sub_gm.graph.inserting_before(output_node):
            for val, ph in zip(out_vals, mut_placeholders):
                sub_gm.graph.call_method("copy_", args=(ph, val))

        output_node.args = (None,)
        sub_gm.recompile()

def test_main():
    # Define two different models to test transformer reusability
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class MulModel(torch.nn.Module):
        def forward(self, x, y): return x * y

    class MatMulModel(torch.nn.Module):
        def forward(self, x, y): return torch.matmul(x, y)

    transformer = OutputAsMutInputsTransformer()

    # Scenario 1: AddModel with non-square inputs (128x64)
    print("\n--- Scenario 1: AddModel (128x64) ---")
    gm1 = fx.symbolic_trace(AddModel())
    inputs1 = [torch.randn(128, 64), torch.randn(128, 64)]
    new_gm1 = transformer(gm1, inputs1)
    print(new_gm1.code)

    # Scenario 2: MulModel (256x256)
    print("\n--- Scenario 2: MulModel (256x256) ---")
    gm2 = fx.symbolic_trace(MulModel())
    inputs2 = [torch.randn(256, 256), torch.randn(256, 256)]
    new_gm2 = transformer(gm2, inputs2)
    print(new_gm2.code)

    # Scenario 3: MatMulModel (128x64 * 64x32) -> Output (128x32)
    print("\n--- Scenario 3: MatMulModel (128x64 * 64x32) ---")
    gm3 = fx.symbolic_trace(MatMulModel())
    inputs3 = [torch.randn(128, 64), torch.randn(64, 32)]
    new_gm3 = transformer(gm3, inputs3)
    print("Generated Code (should use x.shape[0] and y.shape[1]):")
    print(new_gm3.code)
    
    # Scenario 4: Dynamic Execution Verification
    print("\n--- Scenario 4: Dynamic Execution Verification ---")
    test_x = torch.randn(10, 5)
    test_y = torch.randn(10, 5)
    try:
        output = new_gm1(test_x, test_y)
        print(f"Success! Input shape: {test_x.shape}, Output shape: {output.shape}")
        torch.testing.assert_close(output, test_x + test_y)
        print("Numerical correctness verified.")
    except Exception as e:
        print(f"Execution failed: {e}")


if __name__ == "__main__":
    test_main()
