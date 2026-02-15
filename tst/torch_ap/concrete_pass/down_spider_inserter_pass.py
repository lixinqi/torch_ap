import torch
import torch.fx as fx
from typing import List, Tuple, Callable
from tst.torch_ap.spider import down_spider
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace


class DownSpiderInserterPass:
    def __init__(self, input_idx: int | Callable[[], int]):
        if isinstance(input_idx, int):
            self.get_input_idx = lambda: input_idx
        else:
            assert callable(input_idx), f"{input_idx}"
            self.get_input_idx = input_idx

    def __call__(self, gm: fx.GraphModule) -> PassResult:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

        input_idx = self.get_input_idx()

        if input_idx < 0 or input_idx >= len(placeholders):
            return PassResult(gm, False)

        # print(f"\n[Before] Target Input Index: {input_idx}")
        # gm.graph.print_tabular()

        target = placeholders[input_idx]
        with gm.graph.inserting_after(target):
            new_node = gm.graph.call_function(down_spider, (target,))
            target.replace_all_uses_with(
                new_node, delete_user_cb=lambda u: u != new_node
            )

        # print(f"[After]")
        # gm.graph.print_tabular()

        return PassResult(gm, True)


def main(gms_with_pos: List[Tuple[fx.GraphModule, int]]) -> None:
    # --- AssertOnlyForTest ---
    def get_topo_hash(g):
        return "->".join(
            [
                str(n.target)
                for n in g.graph.nodes
                if n.op in ["call_function", "call_method"]
            ]
        )

    assert len({get_topo_hash(gm) for gm, _ in gms_with_pos}) >= 3
    assert (
        len(
            {
                len([n for n in gm.graph.nodes if n.op == "placeholder"])
                for gm, _ in gms_with_pos
            }
        )
        >= 3
    )
    assert len({idx for _, idx in gms_with_pos}) >= 3

    # --- Inline Logic ---
    for gm, input_idx in gms_with_pos:
        inserter = DownSpiderInserter(input_idx)
        inserter(gm)


def run_pipeline():
    def mk_gm(in_count: int, layers: int):
        args = ", ".join([f"x{i}" for i in range(in_count)])
        ops = "\n    ".join([f"x0 = x0 + {i}" for i in range(layers)])
        code = f"class M(torch.nn.Module):\n  def forward(self, {args}):\n    {ops}\n    return x0"
        loc = {}
        exec(code, globals(), loc)
        return torch_ap_trace(loc["M"]())

    data = [(mk_gm(1, 1), 0), (mk_gm(2, 2), 1), (mk_gm(3, 3), 2)]
    main(data)


if __name__ == "__main__":
    run_pipeline()
