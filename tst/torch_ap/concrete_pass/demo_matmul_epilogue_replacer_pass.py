import torch
import torch.fx as fx
import inspect
import string
from typing import Any
from torch.fx.passes.infra.pass_manager import PassResult
from tst.torch_ap.torch_ap_trace import torch_ap_trace
from typing import Callable, List


def reset_func_arg_names(arg_names):
    # arg_names is a list like ['x', 'y', 'z']
    args_str = ", ".join(arg_names)
    import random

    func_name = "dynamic_func_" + "".join(random.choices(string.ascii_lowercase, k=5))

    source = f"""
def {func_name}(f):
    def func({args_str}):
        return f({args_str})
    return func
"""
    namespace = {}
    exec(source, globals(), namespace)
    return namespace[func_name]


def get_arg_names(func: Callable) -> List[str]:
    """
    Viba: get_arg_names := list[str] <- Callable
    Reflects a callable to extract its parameter names.
    """
    # Introspect the function signature
    signature = inspect.signature(func)

    # Extract names from the mapping of parameters
    return list(signature.parameters.keys())


class DemoMatmulEpilogueReplacerPass:
    """
    DemoMatmulEpilogueReplacerPass :=
        PassResult <- None <- $get_torch_module <- $tracer <- $epilogue_func
    """

    def __init__(self, epilogue_func: Any = None):
        if epilogue_func is None:

            def epilogue_func(x, bias):
                return torch.tanh(x + bias)

        arg_names = get_arg_names(epilogue_func)

        @reset_func_arg_names(["_mm_in0", "_mm_in1", *arg_names[1:]])
        def matmul_plus_epilogue(x: torch.Tensor, y: torch.Tensor, *args):
            out = torch.matmul(x, y)
            return epilogue_func(out, *args)

        self.matmul_plus_epilogue = matmul_plus_epilogue

    def __call__(self, _unused: None) -> PassResult:
        """
        Executes the trace on the synthesized module.
        """
        generated_gm = torch_ap_trace(self.matmul_plus_epilogue)

        # Always returns modified=True as it generates a new GraphModule
        return PassResult(graph_module=generated_gm, modified=True)


# --- Test Environment ---


def main():
    # Instantiate with the epilogue function
    replacer_pass = DemoMatmulEpilogueReplacerPass()

    # Invoke with None as per Viba signature
    result = replacer_pass(None)

    print("--- [Generated Graph: Matmul + Tanh(Square)] ---")
    result.graph_module.graph.print_tabular()

    print("\n--- [Generated Python Code] ---")
    print(result.graph_module.code)


if __name__ == "__main__":
    main()
