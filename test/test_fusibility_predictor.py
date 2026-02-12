import torch

def fusibility_of(epilogue_func) -> bool:
    """Check if an epilogue function is fusible."""
    from torch_ap.torch_ap_trace import torch_ap_trace
    from torch_ap.concrete_pass.demo_matmul_epilogue_replacer_pass import DemoMatmulEpilogueReplacerPass
    from torch_ap.trivial_ops_folder_pass import TrivialOpsFolderPass
    from torch_ap.concrete_pass.matmul_epilogue_util import get_matmul_epilogue_arg_name_to_is_mm_out
    from torch_ap.concrete_pass.matmul_epilogue_extractor_pass import MatmulEpilogueExtractorPass
    from torch_ap.concrete_pass.matmul_epilogue_fusibility_predictor import (
        MatmuEpilogueFusibilityPredicator,
    )

    demo_pass = DemoMatmulEpilogueReplacerPass(epilogue_func)
    gm = torch_ap_trace(epilogue_func)
    res = demo_pass(gm)
    matmul_plus_epilogue = res.graph_module

    res = TrivialOpsFolderPass()(matmul_plus_epilogue)
    matmul_plus_epilogue = res.graph_module

    arg_list = get_matmul_epilogue_arg_name_to_is_mm_out(matmul_plus_epilogue)
    mm_idx_list = [i for i, (_, is_mm) in enumerate(arg_list) if is_mm]
    if not mm_idx_list:
        raise ValueError("No matmul output found")
    mm_idx = mm_idx_list[0]

    epilogue_gm = MatmulEpilogueExtractorPass()(matmul_plus_epilogue).graph_module

    # Default config - no custom overrides needed
    predictor = MatmuEpilogueFusibilityPredicator(config_pattern_rewriters=lambda x: x)

    return predictor(epilogue_gm, mm_idx)


def test_case_1():
    """Test Case 1: Fixed-point reduction (tanh -> relu -> reduction)"""
    print("=" * 60)
    print("Test Case 1: Algebraic Fixed-Point Loop (tanh -> relu -> reduction)")
    print("=" * 60)

    def case_math(x, w):
        return torch.tanh(x**2) + w

    result = fusibility_of(case_math)

    print(f"\nFusibility Prediction: {result}")
    return result


def test_case_2():
    """Test Case 2: Multi-load US fusion"""
    print("=" * 60)
    print("Test Case 2: Multi-parameter Topology Fusion")
    print("=" * 60)

    def case_topo(x, w1, w2):
        return x + w1 + w2

    result = fusibility_of(case_topo)

    print(f"\nFusibility Prediction: {result}")
    return result


def test_case_3():
    """Test Case 3: Tuple Output"""
    print("=" * 60)
    print("Test Case 3: Tuple Return Indexing")
    print("=" * 60)

    def case_tuple(x, w):
        return torch.relu(x) + w, x

    result = fusibility_of(case_tuple)

    print(f"\nFusibility Prediction: {result}")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MatmuEpilogueFusibilityPredicator Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("Case 1: tanh(x**2) + w", test_case_1()))
    results.append(("Case 2: x + w1 + w2", test_case_2()))
    results.append(("Case 3: Tuple Output", test_case_3()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
