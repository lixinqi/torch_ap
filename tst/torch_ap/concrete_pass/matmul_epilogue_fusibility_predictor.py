from typing import Callable, List, NamedTuple, Optional, Union
import copy
import torch
import torch.fx as fx

from torch_ap.load_op_inserter_pass import LoadOpInserterPass
from torch_ap.store_op_inserter_pass import StoreOpInserterPass
from torch_ap.pattern_replacer_pass import PatternReplacerPass
from torch_ap.concrete_pass.down_spider_inserter_pass import DownSpiderInserterPass
from torch_ap.spider import down_spider as DS, up_spider as US
from torch_ap.load_store_op import load, store


# TorchFunction := (torch.Tensor | List[torch.Tensor]) <- VariadicList[torch.Tensor]
TorchFunction = Callable[..., Union[torch.Tensor, List[torch.Tensor]]]


class AccessTopoRule(NamedTuple):
    pattern_name: str
    replacement_name: str
    pattern_func: TorchFunction
    replacement_func: TorchFunction


class ConfirmPattern(NamedTuple):
    pattern_name: str
    pattern_func: TorchFunction


class MatmuEpilogueFusibilityPredicator:
    """
    MatmuEpilogueFusibilityPredicator :=
        # __call__
        bool <- $mm_epi fx.GraphModule <- $mm_out_as_epi_in_index int
        # __init__
        <- $config_pattern_rewriters (list[AccessTopoRule] <- list[AccessTopoRule])
        <- $config_pattern_removers (list[ConfirmPattern] <- list[ConfirmPattern])
    """

    def __init__(
        self,
        config_pattern_rewriters: Callable[[List[AccessTopoRule]], List[AccessTopoRule]],
        config_pattern_removers: Optional[Callable[[List[ConfirmPattern]], List[ConfirmPattern]]] = None,
    ):
        self._config_pattern_rewriters = config_pattern_rewriters
        self._config_pattern_remover_overrider = config_pattern_removers

    @property
    def config_pattern_rewriters(self) -> List[AccessTopoRule]:
        return self._config_pattern_rewriters(self._default_rewriters())

    def _default_rewriters(self) -> List[AccessTopoRule]:
        return [
            AccessTopoRule("y=x**2", "y=relu(x)", lambda x: x**2, lambda x: torch.relu(x)),
            AccessTopoRule("y=tanh(x)", "y=relu(x)", lambda x: torch.tanh(x), lambda x: torch.relu(x)),
            AccessTopoRule("z=DS(x)+y", "z=DS(US(x,y))", lambda x, y: DS(x) + y, lambda x, y: DS(US(x, y))),
            AccessTopoRule("z=y+DS(x)", "z=DS(US(x,y))", lambda x, y: y + DS(x), lambda x, y: DS(US(x, y))),
            AccessTopoRule("z=relu(DS(x))", "z=DS(x)", lambda x: torch.relu(DS(x)), lambda x: DS(x)),
        ]

    def _default_removers(
        self,
        mm_epi: fx.GraphModule,
        mm_out_idx: int
    ) -> List[ConfirmPattern]:
        """Generate default ConfirmPatterns from epilogue structure."""
        arg_list = self._get_arg_list(mm_epi)
        output_indices = self._get_output_indices(mm_epi)

        mm_out_arg_names = [name for name, is_mm in arg_list if is_mm]
        other_arg_names = [name for name, is_mm in arg_list if not is_mm]

        removers: List[ConfirmPattern] = []

        for out_idx in output_indices:
            removers.append(ConfirmPattern(
                f"store(DS(x, {out_idx}))",
                lambda x, idx=out_idx: store(DS(x), idx)
            ))

        for arg_name in other_arg_names:
            removers.append(ConfirmPattern(
                f"US(x, load(y, {arg_name}))",
                lambda x, y, n=arg_name: US(x, load(y, n))
            ))

        if mm_out_arg_names:
            mm_arg_name = mm_out_arg_names[0]
            removers.append(ConfirmPattern(
                f"load(x, {mm_arg_name})",
                lambda x, n=mm_arg_name: load(x, n)
            ))

        return removers

    def _get_arg_list(self, gm: fx.GraphModule) -> List[tuple[str, bool]]:
        """Get (name, is_mm_output) for each placeholder."""
        placeholder_names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        if len(placeholder_names) == 0:
            return []
        return [(name, i == len(placeholder_names) - 1) for i, name in enumerate(placeholder_names)]

    def _get_output_indices(self, gm: fx.GraphModule) -> List[Optional[int]]:
        out_node = next(n for n in gm.graph.nodes if n.op == "output")
        res = out_node.args[0]
        if isinstance(res, (tuple, list)):
            return list(range(len(res)))
        return [None]

    def __call__(
        self,
        mm_epi: fx.GraphModule,
        mm_out_as_epi_in_index: int
    ) -> bool:
        """
        Predict if the matmul epilogue is fusible.

        Logic: If all confirm patterns can be successfully removed,
        the graph is fusible.
        """
        working_gm = copy.deepcopy(mm_epi)

        # Step 1: Insert DS/Load/Store
        working_gm = self._insert_spiders(working_gm, mm_out_as_epi_in_index)

        # Step 2: Apply rewriters in fixed-point loop
        working_gm = self._fixed_point_loop(working_gm)

        # Step 3: Get removers (default or custom)
        removers = self._default_removers(mm_epi, mm_out_as_epi_in_index)
        if self._config_pattern_remover_overrider:
            removers = self._config_pattern_remover_overrider(removers)

        # Step 4: Check confirm patterns by attempting to remove them
        from torch.fx import subgraph_rewriter
        from torch_ap.torch_ap_trace import torch_ap_trace

        for remover in removers:
            pattern_gm = torch_ap_trace(remover.pattern_func)
            replacement_gm = torch_ap_trace(remover.pattern_func)  # Identity replacement

            matches = subgraph_rewriter.replace_pattern(
                working_gm, pattern_gm, replacement_gm
            )

            if len(matches) == 0:
                # Pattern could not be removed - not fusible
                return False

        # All patterns removed - fusible
        return True

    def _insert_spiders(self, gm: fx.GraphModule, mm_out_idx: int) -> fx.GraphModule:
        res = DownSpiderInserterPass(input_idx=mm_out_idx)(gm)
        gm = res.graph_module
        res = LoadOpInserterPass()(gm)
        gm = res.graph_module
        res = StoreOpInserterPass()(gm)
        return res.graph_module

    def _fixed_point_loop(self, gm: fx.GraphModule) -> fx.GraphModule:
        curr_gm = gm
        while True:
            any_mod = False
            for rule in self.config_pattern_rewriters:
                replacer = PatternReplacerPass(rule.pattern_func, rule.replacement_func)
                res = replacer(curr_gm)
                if res.modified:
                    any_mod = True
                    curr_gm = res.graph_module
            if not any_mod:
                break
        return curr_gm
