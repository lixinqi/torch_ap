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

        new_graph = new_gm.graph
        placeholder_nodes = []
        for n in target.graph.nodes:
            if n.op == "placeholder":
                new_ph = new_graph.placeholder(n.target)
                placeholder_nodes.append(new_ph)

        new_graph.call_module("sub", args=tuple(placeholder_nodes))
        new_gm.recompile()
        return new_gm, placeholder_nodes

    def _extract_input_symbols(
        self, 
        fake_input: torch.Tensor, 
        ph_node: fx.Node, 
        symbol_map: Dict[Any, Tuple[fx.Node, int]]
    ):
        """Helper to map SymInts from a single input to their source node and index."""
        for i, dim in enumerate(fake_input.shape):
            if isinstance(dim, torch.SymInt):
                key = dim.node if hasattr(dim, "node") else str(dim)
                symbol_map[key] = (ph_node, i)

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
            
            # Build the map using a helper to reduce nesting
            for fake_input, ph_node in zip(fake_inputs, placeholders):
                self._extract_input_symbols(fake_input, ph_node, input_symbol_map)
            
            with torch.no_grad():
                outputs = target(*fake_inputs)
            
            symbolic_shapes = [list(o.shape) for o in (outputs if isinstance(outputs, (tuple, list)) else [outputs])]
                
        return symbolic_shapes, input_symbol_map

    def _bind_single_shape(
        self, 
        shape: List[torch.SymInt], 
        input_symbol_map: Dict[Any, Tuple[fx.Node, int]]
    ) -> List[Union[int, Tuple[fx.Node, int]]]:
        """Helper to bind a single shape's SymInts to their sources."""
        recipe = []
        for dim in shape:
            if isinstance(dim, int):
                recipe.append(dim)
            elif isinstance(dim, torch.SymInt):
                key = dim.node if hasattr(dim, "node") else str(dim)
                recipe.append(input_symbol_map.get(key, dim))
            else:
                recipe.append(dim)
        return recipe

    def _bind_symbols_to_nodes(
        self, 
        symbolic_shapes: List[List[torch.SymInt]], 
        input_symbol_map: Dict[Any, Tuple[fx.Node, int]]
    ) -> List[List[Union[int, Tuple[fx.Node, int]]]]:
        """
        3. ($runnable_output_shapes <- $symbolic_output_shapes <- $input_symbol_map)
        Uses helpers to reduce nesting while finding the EXACT source for each SymInt.
        """
        return [self._bind_single_shape(shape, input_symbol_map) for shape in symbolic_shapes]

    def _materialize_shape(
        self, 
        gm: fx.GraphModule, 
        recipe: List[Union[int, Tuple[fx.Node, int]]],
        cache: Dict[fx.Node, fx.Node]
    ) -> Tuple[Any, ...]:
        """Helper to convert a shape recipe into a tuple of FX nodes/ints."""
        actual_shape_nodes = []
        for item in recipe:
            if isinstance(item, int):
                actual_shape_nodes.append(item)
            elif isinstance(item, tuple):
                source_node, dim_idx = item
                if source_node not in cache:
                    cache[source_node] = gm.graph.call_function(getattr, args=(source_node, "shape"))
                shape_attr = cache[source_node]
                dim_val = gm.graph.call_function(operator.getitem, args=(shape_attr, dim_idx))
                actual_shape_nodes.append(dim_val)
            else:
                actual_shape_nodes.append(item)
        return tuple(actual_shape_nodes)

    def _insert_empty_nodes(
        self, 
        gm: fx.GraphModule, 
        runnable_shapes: List[List[Union[int, Tuple[fx.Node, int]]]], 
        dtypes: List[torch.dtype],
        placeholders: List[fx.Node]
    ) -> List[fx.Node]:
        """
        4. ($inserted_mut_input_nodes <- $gm_with_sub <- $runnable_output_shapes)
        Materializes the recipe into the FX graph.
        """
        insert_point = next(n for n in gm.graph.nodes if n.op != "placeholder")
        inserted_empty_nodes = []
        shape_node_cache = {} # Cache for getattr(node, 'shape')
        
        with gm.graph.inserting_before(insert_point):
            for recipe in runnable_shapes:
                # Materialize the shape recipe into actual FX nodes
                actual_shape = self._materialize_shape(gm, recipe, shape_node_cache)
                
                empty_node = gm.graph.call_function(
                    torch.empty, 
                    args=(actual_shape,), 
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


def run_dynamic_test(name, model, example_inputs, dynamic_test_inputs):
    print(f"\n=== {name} ===")

    transformer = OutputAsMutInputsTransformer()
    gm = fx.symbolic_trace(model)
    new_gm = transformer(gm, example_inputs)

    print("\n--- Transformed Code ---")
    print(new_gm.code)

    print("\n--- Dynamic Shape Tests ---")

    for inputs in dynamic_test_inputs:
        shapes = [tuple(t.shape) for t in inputs]
        try:
            out = new_gm(*inputs)
            ref = model(*inputs)

            correct = torch.allclose(out, ref)

            print(f"✅ Input shapes {shapes} "
                  f"→ Output shape {tuple(out.shape)} "
                  f"| Correct: {correct}")

        except Exception as e:
            print(f"❌ Input shapes {shapes} FAILED")
            print(f"   Error: {e}")


def test_main():

    class AddModel(torch.nn.Module):
        def forward(self, x, y): return x + y

    class MulModel(torch.nn.Module):
        def forward(self, x, y): return x * y

    class MatMulModel(torch.nn.Module):
        def forward(self, x, y): return torch.matmul(x, y)

    # -------------------------------------------------
    # Scenario 1: AddModel
    # -------------------------------------------------
    run_dynamic_test(
        name="Scenario 1: AddModel (base: 128x64)",
        model=AddModel(),
        example_inputs=[
            torch.randn(128, 64),
            torch.randn(128, 64),
        ],
        dynamic_test_inputs=[
            (torch.randn(128, 64), torch.randn(128, 64)),  # original
            (torch.randn(256, 64), torch.randn(256, 64)),  # change batch
            (torch.randn(32, 64), torch.randn(32, 64)),    # smaller batch
            (torch.randn(128, 128), torch.randn(128, 128)) # change feature dim
        ]
    )

    # -------------------------------------------------
    # Scenario 2: MulModel
    # -------------------------------------------------
    run_dynamic_test(
        name="Scenario 2: MulModel (base: 256x256)",
        model=MulModel(),
        example_inputs=[
            torch.randn(256, 256),
            torch.randn(256, 256),
        ],
        dynamic_test_inputs=[
            (torch.randn(256, 256), torch.randn(256, 256)),
            (torch.randn(128, 256), torch.randn(128, 256)),
            (torch.randn(256, 128), torch.randn(256, 128)),
        ]
    )

    # -------------------------------------------------
    # Scenario 3: MatMulModel
    # -------------------------------------------------
    run_dynamic_test(
        name="Scenario 3: MatMulModel (base: 128x64 @ 64x32)",
        model=MatMulModel(),
        example_inputs=[
            torch.randn(128, 64),
            torch.randn(64, 32),
        ],
        dynamic_test_inputs=[
            (torch.randn(128, 64), torch.randn(64, 32)),
            (torch.randn(256, 64), torch.randn(64, 32)),   # change batch
            (torch.randn(128, 128), torch.randn(128, 16)), # change inner dims
        ]
    )


if __name__ == "__main__":
    test_main()
