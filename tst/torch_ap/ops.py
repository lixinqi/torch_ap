import torch
import torch.fx as fx


def down_spider(x):
    return torch.relu(x)


def up_spider(x, y):
    return y


def load(x, placeholder_name: str):
    return torch.relu(x)


def store(x, output_idx: int | None):
    return torch.relu(x)


atom_funcs = (down_spider, up_spider, load, store)
