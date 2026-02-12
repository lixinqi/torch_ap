import torch
import torch.fx as fx
from typing import Callable, List, Dict, Any, Union, Optional, Tuple

# --- Imports (No local definitions or decorators, just use) ---
from tst.torch_ap.torch_ap_trace import torch_ap_trace
from tst.torch_ap.spider import down_spider as DS, up_spider as US
from tst.torch_ap.load_store_op import load, store
from torch.fx.passes.infra.pass_manager import PassManager, PassResult
from tst.torch_ap.concrete_pass.demo_matmul_epilogue_replacer_pass import (
    DemoMatmulEpilogueReplacerPass,
)
from tst.torch_ap.trivial_ops_folder_pass import TrivialOpsFolderPass
from tst.torch_ap.concrete_pass.matmul_epilogue_extractor_pass import (
    MatmulEpilogueExtractorPass,
)
from tst.torch_ap.load_op_inserter_pass import LoadOpInserterPass
from tst.torch_ap.store_op_inserter_pass import StoreOpInserterPass
from tst.torch_ap.concrete_pass.down_spider_inserter_pass import DownSpiderInserterPass
from tst.torch_ap.pattern_replacer_pass import PatternReplacerPass
from tst.torch_ap.pattern_remover_pass import PatternRemoverPass
from tst.torch_ap.concrete_pass.matmul_epilogue_util import (
    get_matmul_epilogue_arg_name_to_is_mm_out,
)


class GraphFusibilityDemo:
    def __init__(self):
        pass

    def __call__(self, epilogue_func: Callable) -> fx.GraphModule:
        # Trace and initial Compose[DemoMatmulEpilogueReplacerPass, TrivialOpsFolderPass]
        gm = torch_ap_trace(epilogue_func)
        setup_pipe = [
            DemoMatmulEpilogueReplacerPass(epilogue_func),
            TrivialOpsFolderPass(),
        ]
        matmul_plus_epilogue = self._run_sequence(gm, setup_pipe)

        # match[list[($epilogue_arg_name str, $is_mm_out bool)]]
        # Now directly a list, no .items() needed
        arg_list = get_matmul_epilogue_arg_name_to_is_mm_out(matmul_plus_epilogue)

        # match[$epilogue_mm_in_arg]
        mm_in_arg = [name for name, is_mm in arg_list if is_mm][0]
        # match[$epilogue_other_args]
        other_args = [name for name, is_mm in arg_list if not is_mm]
        # match[$mm_out_as_epi_input_idx]
        mm_idx = [i for i, (name, is_mm) in enumerate(arg_list) if is_mm][0]

        # match[$epilogue_gm]
        epilogue_gm = MatmulEpilogueExtractorPass()(matmul_plus_epilogue).graph_module

        # Run main optimization sequence
        opt_pipeline = self._assemble_opt_pipeline(
            epilogue_gm, mm_in_arg, other_args, mm_idx
        )
        return self._run_sequence(matmul_plus_epilogue, opt_pipeline)

    def _assemble_opt_pipeline(self, epilogue_gm, mm_in, others, mm_idx) -> List[Any]:
        """Flattened pipeline assembly."""
        insertion_passes = [
            MatmulEpilogueExtractorPass(),
            DownSpiderInserterPass(input_idx=mm_idx),
            LoadOpInserterPass(),
            StoreOpInserterPass(),
        ]

        # AccessTopoReduction definitions
        replacements = [
            ("y=x**2", "y=relu(x)", lambda x: x**2, lambda x: torch.relu(x)),
            (
                "y=tanh(x)",
                "y=relu(x)",
                lambda x: torch.tanh(x),
                lambda x: torch.relu(x),
            ),
            (
                "z=DS(x)+y",
                "z=DS(US(x,y))",
                lambda x, y: DS(x) + y,
                lambda x, y: DS(US(x, y)),
            ),
            (
                "z=y+DS(x)",
                "z=DS(US(x,y))",
                lambda x, y: y + DS(x),
                lambda x, y: DS(US(x, y)),
            ),
            ("z=relu(DS(x))", "z=DS(x)", lambda x: torch.relu(DS(x)), lambda x: DS(x)),
        ]

        # Removal match-loops
        removals = []
        for out_idx in self._get_output_indices(epilogue_gm):
            removals.append(
                self._remover(
                    f"store(DS(x, {out_idx}))", lambda x: store(DS(x), out_idx)
                )
            )

        for load_name in others:
            removals.append(
                self._remover(
                    f"US(x, load(y, {load_name}))",
                    lambda x, y, n=load_name: US(x, load(y, n)),
                )
            )

        removals.append(self._remover(f"load(x, {mm_in})", lambda x: load(x, mm_in)))

        return insertion_passes + [("LOOP_REPLACE", replacements)] + removals

    def _get_output_indices(self, gm: fx.GraphModule) -> List[Optional[int]]:
        """Handles Viba Oneof logic for graph outputs."""
        out_node = next(n for n in gm.graph.nodes if n.op == "output")
        res = out_node.args[0]
        return list(range(len(res))) if isinstance(res, (tuple, list)) else [None]

    def _remover(self, p_str: str, p_func: Callable) -> PatternRemoverPass:
        p = PatternRemoverPass(p_func)
        p.viba_pattern_str = p_str  # Meta for logging
        return p

    def _run_sequence(self, gm: fx.GraphModule, pipeline: List[Any]) -> fx.GraphModule:
        curr_gm = gm
        for entry in pipeline:
            if isinstance(entry, tuple) and entry[0] == "LOOP_REPLACE":
                curr_gm = self._fixed_point_loop(curr_gm, entry[1])
            else:
                curr_gm = self._execute_pass(curr_gm, entry)
        return curr_gm

    def _fixed_point_loop(
        self, gm: fx.GraphModule, defs: List[tuple]
    ) -> fx.GraphModule:
        """$loop_transform_until_no_pass_matched."""
        curr_gm = gm
        while True:
            any_mod = False
            for p_str, r_str, p_f, r_f in defs:
                replacer = PatternReplacerPass(p_f, r_f)
                print(f"[Transform] Replacing: {p_str} -> {r_str}")
                res = replacer(curr_gm)
                if res.modified:
                    any_mod = True
                    curr_gm = res.graph_module
                    print(f"--- Code after replacing ({p_str}) ---\n{curr_gm.code}")
            if not any_mod:
                break
        return curr_gm

    def _execute_pass(self, gm: fx.GraphModule, p: Any) -> fx.GraphModule:
        if hasattr(p, "viba_pattern_str"):
            print(f"[Transform] Removing: {p.viba_pattern_str}")
        res = p(gm)
        print(f"--- After {type(p).__name__} ---\n{res.graph_module.code}")
        return res.graph_module


# --- Test Suite ---


def test_main():
    demo = GraphFusibilityDemo()
    RED, RESET = "\033[91m", "\033[0m"

    # Case 1: Fixed-point reduction
    print(
        f"{RED}Test Case 1: Algebraic Fixed-Point Loop (tanh -> relu -> reduction){RESET}"
    )

    def case_math(x, w):
        return torch.tanh(x**2) + w

    demo(case_math)

    # Case 2: Multi-load US fusion
    print(f"\n{RED}Test Case 2: Multi-parameter Topology Fusion{RESET}")

    def case_topo(x, w1, w2):
        return x + w1 + w2

    demo(case_topo)

    # Case 3: Tuple Output
    print(f"\n{RED}Test Case 3: Tuple Return Indexing{RESET}")

    def case_tuple(x, w):
        return torch.relu(x) + w, x

    demo(case_tuple)


if __name__ == "__main__":
    test_main()
