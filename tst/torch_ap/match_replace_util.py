import torch
import torch.fx as fx
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional
from tst.torch_ap.torch_ap_trace import torch_ap_trace


@dataclass
class MatchContext:
    nodes_map: dict[fx.Node, fx.Node]
    target: fx.GraphModule
    pattern: fx.GraphModule


def fx_graph_match_first_pattern(
    target: fx.GraphModule,
    get_pattern: Callable[[], fx.GraphModule],
    extra_check: Callable[[Dict[fx.Node, fx.Node]], bool],
) -> MatchContext | None:
    """Find and replace pattern with I/O consistency enforcement."""
    pattern = get_pattern()
    p_nodes = [n for n in pattern.graph.nodes if n.op not in ["placeholder", "output"]]
    t_nodes = [n for n in target.graph.nodes if n.op not in ["placeholder", "output"]]

    # 1. Matching
    match_result = None
    for t_node in t_nodes:
        res = try_match_at(t_node, p_nodes, pattern)
        match_ctx = MatchContext(nodes_map=res, target=target, pattern=pattern)
        if res and extra_check(match_ctx):
            match_result = res
            break

    if not match_result:
        return None

    match_ctx = MatchContext(match_result, target=target, pattern=pattern)
    return match_ctx


def fx_graph_replace_first_pattern(
    target: fx.GraphModule,
    get_pattern: Callable[[], fx.GraphModule],
    extra_check: Callable[[Dict[fx.Node, fx.Node]], bool],
    replacement_gen: Callable[[Any], fx.GraphModule],
) -> (fx.GraphModule, bool):
    """Find and replace pattern with I/O consistency enforcement."""
    pattern = get_pattern()
    p_nodes = [n for n in pattern.graph.nodes if n.op not in ["placeholder", "output"]]
    t_nodes = [n for n in target.graph.nodes if n.op not in ["placeholder", "output"]]

    # 1. Matching
    match_ctx = fx_graph_match_first_pattern(
        target=target,
        get_pattern=lambda: pattern,
        extra_check=extra_check,
    )
    if match_ctx is None:
        return target, False
    replacement = replacement_gen(match_ctx)

    p_in, p_out = get_io_count(pattern)
    r_in, r_out = get_io_count(replacement)

    # Enforce protocol assertions
    assert p_in == r_in, f"Input mismatch: pattern({p_in}) vs replacement({r_in})"
    assert p_out == r_out, f"Output mismatch: pattern({p_out}) vs replacement({r_out})"

    # 3. Surgery
    match_result = match_ctx.nodes_map
    p_output_node = next(n for n in pattern.graph.nodes if n.op == "output")
    t_exit_node = match_result[p_output_node.args[0]]
    r_placeholders = [n for n in replacement.graph.nodes if n.op == "placeholder"]
    p_placeholders = [n for n in pattern.graph.nodes if n.op == "placeholder"]

    target.graph.inserting_before(t_exit_node)
    node_map = {}

    for i, r_ph in enumerate(r_placeholders):
        node_map[r_ph] = match_result[p_placeholders[i]]

    for r_node in replacement.graph.nodes:
        if r_node.op == "placeholder":
            continue
        if r_node.op == "output":
            t_exit_node.replace_all_uses_with(node_map[r_node.args[0]])
            continue
        node_map[r_node] = target.graph.node_copy(r_node, lambda n: node_map[n])

    # 4. Cleanup
    for p_node in reversed(p_nodes):
        target.graph.erase_node(match_result[p_node])

    target.graph.lint()
    target.recompile()
    return target, True


def get_io_count(gm: fx.GraphModule):
    """Returns counts for (placeholders, outputs)."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    # FX output nodes typically have one 'output' node whose args[0] is the return value(s)
    outputs = [n for n in gm.graph.nodes if n.op == "output"]
    return len(placeholders), len(outputs)


def check_node_match(pn: fx.Node, tn: fx.Node) -> bool:
    """Check if a pattern node and target node are semantically equivalent."""
    if pn.op != tn.op:
        return False

    if pn.op == "call_function":
        p_target = str(pn.target).split(" at ")[0]
        t_target = str(tn.target).split(" at ")[0]
        return p_target == t_target

    return True


def try_match_at(
    t_start_node: fx.Node, p_nodes: List[fx.Node], pattern: fx.GraphModule
) -> Optional[Dict[fx.Node, fx.Node]]:
    """Attempt to match a subgraph starting from a specific target node."""
    mapping = {}
    curr_t = t_start_node

    for p_node in p_nodes:
        if curr_t is None or not check_node_match(p_node, curr_t):
            return None
        mapping[p_node] = curr_t

        if p_node == p_nodes[-1]:
            continue
        if not curr_t.users:
            return None
        curr_t = list(curr_t.users.keys())[0]

    p_placeholders = [n for n in pattern.graph.nodes if n.op == "placeholder"]
    t_entry = mapping[p_nodes[0]]
    for i, p_ph in enumerate(p_placeholders):
        mapping[p_ph] = t_entry.args[i] if i < len(t_entry.args) else None

    return mapping


# --- Test Data Implementation ---

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

    def get_pattern():
        class P(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = PatternModule()

            def forward(self, x, y):
                return self.mod(torch.matmul(x, y))

        return fx.GraphModule(P(), tracer.trace(P()))

    def replacement_gen(ctx):
        t_node = next(tn for pn, tn in ctx.nodes_map.items() if pn.op == "call_module")
        bias_val = t_gm.get_submodule(t_node.target).bias

        class Replacement(torch.nn.Module):
            def forward(self, x, y):
                # replacement contains 0 call_module nodes
                return torch.matmul(x, y) + bias_val

        return torch_ap_trace(Replacement())

    # --- Verification ---
    print("--- Before Transformation ---")
    print(t_gm.code)

    # AssertOnlyForTest conditions check
    pattern_gm = get_pattern()
    assert len(t_gm.graph.nodes) >= 1
    assert len(pattern_gm.graph.nodes) >= 1
    assert any(n.op == "call_module" for n in t_gm.graph.nodes)
    assert any(n.op == "call_module" for n in pattern_gm.graph.nodes)

    result_gm, _ = fx_graph_replace_first_pattern(
        t_gm, get_pattern, lambda m: True, replacement_gen
    )

    print("\n--- After Transformation ---")
    print(result_gm.code)

    # Final assertion: NumOfFilteredOps[replacement, "call_module"] == 0
    final_mods = [n for n in result_gm.graph.nodes if n.op == "call_module"]
    assert len(final_mods) == 0
    print(f"\nRemaining call_module: {len(final_mods)}")
    print("Success: Transformation consistent with input/output protocol.")
