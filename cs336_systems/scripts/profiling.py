import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from typing import Callable

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy


model_sizes = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
}

def run_LM(
        model_size: str,
        context_length: int,
        num_warmups: int = 5,
        num_steps: int = 10
):
    """
    运行模型
    """
    # 默认参数
    vocab_size = 10000
    rope_theta = 10000.0
    batch_size = 4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_path = "data/token/TinyStories_valid_10000_token_ids.npy"

    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_sizes[model_size]["d_model"],
        num_layers=model_sizes[model_size]["num_layers"],
        num_heads=model_sizes[model_size]["num_heads"],
        d_ff=model_sizes[model_size]["d_ff"],
        rope_theta=rope_theta
    ).to(device=device)

    # 获取输入batch
    dataset = np.load(dataset_path, mmap_mode='r+')
    inputs, targets = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )

    for step in range(num_warmups):
        with nvtx.range(f"Warmup Step {step}"):
            with nvtx.range("Forward Pass"):
                logits = model(inputs)
            with nvtx.range("Cross Entropy"):
                loss = cross_entropy(logits, targets)
            with nvtx.range("Backward Pass"):
                loss.backward()

    for step in range(num_steps):
        with nvtx.range(f"Step {step}"):
            with nvtx.range("Forward Pass"):
                logits = model(inputs)
            with nvtx.range("Cross Entropy"):
                loss = cross_entropy(logits, targets)
            with nvtx.range("Backward Pass"):
                loss.backward()



if __name__ == "__main__":
    run_LM(
        model_size="small",
        context_length=256,
        num_warmups=5,
        num_steps=10
    )