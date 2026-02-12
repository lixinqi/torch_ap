import torch
import torch.fx as fx
import networkx as nx
from typing import Any
from tst.torch_ap.spider import up_spider, down_spider


class ReductionRuleGraphValidator:
    """
    ReductionRuleGraphValidator := void <- $p_func <- $r_func <- ()
    """

    def __init__(self):
        # [Viba Contract] spiders are leaf nodes for this tracer instance
        self.tracer = fx.Tracer(autowrap_functions=(up_spider, down_spider))

    def __call__(self, pattern_func: Any, replacement_func: Any) -> None:
        p_gm = fx.GraphModule(torch.nn.Module(), self.tracer.trace(pattern_func))
        r_gm = fx.GraphModule(torch.nn.Module(), self.tracer.trace(replacement_func))

        # 1. Assert Placeholder/Output parity
        get_phs = lambda gm: [n for n in gm.graph.nodes if n.op == "placeholder"]

        def get_out_count(gm):
            out = next(n for n in gm.graph.nodes if n.op == "output")
            args = out.args[0]
            if args is None:
                return 0
            return len(args) if isinstance(args, (tuple, list)) else 1

        assert len(get_phs(p_gm)) == len(get_phs(r_gm)), "Placeholder count mismatch"
        assert get_out_count(p_gm) == get_out_count(r_gm), "Output count mismatch"

        # 2. Assert Spiders > 0 (Semantic constraint)
        spider_targets = {up_spider, down_spider}
        num_spiders = len(
            [
                n
                for n in p_gm.graph.nodes
                if n.op == "call_function" and n.target in spider_targets
            ]
        )
        assert num_spiders > 0, "Pattern must contain at least one spider op"

        # 3. Assert Weakly Connected Components == 1
        def get_wcc(gm):
            g = nx.DiGraph()
            # Include placeholders to act as the "bus" connecting independent ops
            # Exclude output node to focus on the computational "body"
            nodes = [n for n in gm.graph.nodes if n.op != "output"]

            # Identity case check
            if not any(n.op not in ("placeholder", "output") for n in gm.graph.nodes):
                return 1

            for n in nodes:
                g.add_node(n.name)
                for arg in n.args:
                    if isinstance(arg, fx.Node) and arg.op != "output":
                        g.add_edge(arg.name, n.name)

            # Remove unused placeholders (they don't count toward fragmentation)
            unused_phs = [n for n, d in g.degree() if d == 0]
            g.remove_nodes_from(unused_phs)

            return nx.number_weakly_connected_components(g)

        assert get_wcc(p_gm) == 1, f"Pattern fragmented ({get_wcc(p_gm)} components)"
        assert (
            get_wcc(r_gm) == 1
        ), f"Replacement fragmented ({get_wcc(r_gm)} components)"


# --- Final 9 Test Case Suite ---


def run_tests():
    v = ReductionRuleGraphValidator()

    # Valid Cases
    def v1_p(x):
        return down_spider(x)

    def v1_r(x):
        return x

    def v2_p(x, y):
        up_spider(x, y)  # Side-effect spider
        return x + y

    def v2_r(x, y):
        return x + y

    def v3_p(x):
        return torch.relu(down_spider(x))

    def v3_r(x):
        return torch.relu(x)

    def v4_p(x):
        return down_spider(x), x

    def v4_r(x):
        return x, x

    # Invalid Cases
    def i1_p(x):
        return torch.relu(x)  # Missing spider

    def i1_r(x):
        return x

    def i2_p(x, y):
        return down_spider(x)  # Arg Drop (p:2, r:1)

    def i2_r(x):
        return x

    def i3_p(x):
        return down_spider(x)

    def i3_r(x):
        a = x + 1
        b = torch.ones_like(x)  # b is a separate island unrelated to 'a' or output
        return a

    def i4_p(x, y):
        a = down_spider(x)
        b = down_spider(y)  # No common inputs or shared ops
        return a

    def i4_r(x, y):
        return x

    def i5_p(x):
        return down_spider(x)

    def i5_r(x):
        return x, x  # Return arity mismatch

    test_list = [
        ("Identity Bypass", v1_p, v1_r, True),
        ("Side-effect Spider", v2_p, v2_r, True),
        ("Wrapped Math", v3_p, v3_r, True),
        ("Multi-Output Identity", v4_p, v4_r, True),
        ("No Spider Error", i1_p, i1_r, False),
        ("Arg Count Error", i2_p, i2_r, False),
        ("Frag Replacement Error", i3_p, i3_r, False),
        ("Frag Pattern Error", i4_p, i4_r, False),
        ("Output Count Error", i5_p, i5_r, False),
    ]

    for name, p, r, expected in test_list:
        try:
            v(p, r)
            print(f"[PASS] {name}")
        except AssertionError as e:
            if not expected:
                print(f"[PASS] {name} caught expected: {e}")
            else:
                print(f"[FAIL] {name} raised unexpectedly: {e}")


if __name__ == "__main__":
    run_tests()
