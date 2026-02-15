import torch
import torch.fx as fx
from typing import List, Optional, Tuple
import copy
import sys
import os
import torch_ap.spider_util as spider_util
from torch_ap.spider import up_spider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class IsGraphFusible:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __call__(self, gm: fx.GraphModule) -> bool:
        clone = graph_utils.clone_graph(gm)
        if self.verbose:
            print(f"[IsGraphFusible] Step 1: Cloned graph")
            if self.verbose:
                print("\n[Original Graph]")
                clone.graph.print_tabular()

        with_spider = self.insert_spiders(clone)
        if self.verbose:
            print(f"[IsGraphFusible] Step 2: Inserted spiders")
            if self.verbose:
                print("\n[After Insert Spider]")
                with_spider.graph.print_tabular()

        # Step 3: umprime
        converted = self.convert_op(with_spider)
        if self.verbose:
            print(f"[IsGraphFusible] Step 3: Converted operations")
            if self.verbose:
                print("\n[After Op Convert]")
                converted.graph.print_tabular()

        # Step 4
        simplified = self.convert_pattern_until_fail(converted)
        if self.verbose:
            print(f"[IsGraphFusible] Step 4: Simplified patterns")
            if self.verbose:
                print("\n[After Pattern Simplify]")
                simplified.graph.print_tabular()

        # Step 5
        with_removed_output = self.remove_matched_pattern_to_all_outputs(simplified)
        if self.verbose:
            print(f"[IsGraphFusible] Step 5: Removed output patterns")

        # Step 6
        with_removed_input = self.remove_matched_pattern_from_all_inputs(with_removed_output)
        if self.verbose:
            print(f"[IsGraphFusible] Step 6: Removed input patterns")
            if self.verbose:
                print("\n[After Remove Input Pattern]")
                with_removed_input.graph.print_tabular()

        # Step 7
        is_empty = graph_utils.is_graph_empty(with_removed_input)
        if self.verbose:
            print(f"[IsGraphFusible] Step 7: Graph is empty: {is_empty}")

        return is_empty
