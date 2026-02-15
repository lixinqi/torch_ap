import torch
import torch.fx as fx
from typing import Any, Union, List

# --- Core Functor Implementation ---


class OpConvertionRuleValidator:
    """
    OpConvertionRuleValidator :=
        void <- $pattern_func <- $replacement_func <- ()
    """

    def __init__(self):
        # # inline: establishment of the tracer
        self.tracer = fx.Tracer()

    def __call__(self, pattern_func: Any, replacement_func: Any) -> None:
        # # inline helper: TraceGraphModule
        def trace_graph_module(func) -> fx.GraphModule:
            return fx.GraphModule(torch.nn.Module(), self.tracer.trace(func))

        # # inline helper: GetNumOpsExcludePlaceholderAndOutput
        def get_num_ops(gm: fx.GraphModule) -> int:
            return len(
                [n for n in gm.graph.nodes if n.op not in ("placeholder", "output")]
            )

        # # Assertions based on Viba definitions
        pattern_gm = trace_graph_module(pattern_func)
        replacement_gm = trace_graph_module(replacement_func)

        # Assert[GetNumOpsExcludePlaceholderAndOutput[...] == 1]
        pattern_ops = get_num_ops(pattern_gm)
        assert (
            pattern_ops == 1
        ), f"Pattern must contain exactly 1 op, found {pattern_ops}"

        replacement_ops = get_num_ops(replacement_gm)
        assert (
            replacement_ops == 1
        ), f"Replacement must contain exactly 1 op, found {replacement_ops}"


# --- Test Environment ---


def main():
    import operator

    # Valid: 1 op (pow)
    def valid_pattern(x):
        return operator.pow(x, 2)

    # Valid: 1 op (mul)
    def valid_replacement(x):
        return x * x

    # Invalid: 2 ops (pow + add)
    def invalid_pattern(x):
        return operator.pow(x, 2) + 1

    validator = OpConvertionRuleValidator()

    print("Checking valid pair...")
    validator(valid_pattern, valid_replacement)
    print("Verification successful.")

    try:
        print("\nChecking invalid pattern...")
        validator(invalid_pattern, valid_replacement)
    except AssertionError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    main()
