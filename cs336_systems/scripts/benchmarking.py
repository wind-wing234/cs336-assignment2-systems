from statistics import mean
from typing import Callable
import numpy as np
import torch
from torch import nn
import timeit
import os
import pandas as pd

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy


def benchmark(description: str, runs: list[Callable], num_warmups: int, num_trials: int):
    """
    测试函数运行时间
    """
    print(f"Benchmarking: {description}")
    # Warmup，不计入时间
    for _ in range(num_warmups):
        for run in runs:
            run()

    # 让CPU和GPU同步，也就是等待上面的warmup完全完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时
    time_results = {run.__name__: [] for run in runs}
    for _ in range(num_trials):
        for run in runs:
            start_time = timeit.default_timer() # returns float seconds
            run()
            # 等待GPU完成计算，再开始下一次run
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            time_results[run.__name__].append(end_time - start_time)
    
    return time_results

def run_LM(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
) -> tuple[Callable, Callable]:
    """
    返回"运行一次前向传播(+反向传播)"的函数
    """
    # 默认参数
    vocab_size = 10000
    context_length = 256
    rope_theta = 10000.0
    batch_size = 4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_path = "data/token/TinyStories_valid_10000_token_ids.npy"

    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
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

    logits: torch.Tensor = None
    def forward():
        nonlocal logits # 修改外部变量要声明
        logits = model(inputs)

    def backward():
        loss = cross_entropy(logits, targets)
        loss.backward()
    
    return forward, backward


def exp1():
    """
    对不同大小的模型进行基准测试，测量前向传递和前向+后向传递的时间
    """
    model_sizes = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
    }
    
    # 基准测试参数
    num_warmups = 5
    num_steps = 10
    
    # 存储结果
    results = []
    
    for size_name, params in model_sizes.items():
        print(f"Testing {size_name} model...")
        
        forward, backward = run_LM(
            d_model=params["d_model"],
            d_ff=params["d_ff"],
            num_layers=params["num_layers"],
            num_heads=params["num_heads"]
        )

        runs = [forward, backward]
        time_results = benchmark(
            description=f"{size_name} model forward and backward",
            runs=runs,
            num_warmups=num_warmups,
            num_trials=num_steps
        )

        forward_times = time_results[forward.__name__]
        backward_times = time_results[backward.__name__]
        
        # 计算统计数据
        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)
        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)
        
        # 添加到结果中
        results.append({
            "Model Size": size_name,
            "d_model": params["d_model"],
            "d_ff": params["d_ff"],
            "num_layers": params["num_layers"],
            "num_heads": params["num_heads"],
            "Forward Mean (s)": forward_mean,
            "Forward Std (s)": forward_std,
            "Backward Mean (s)": backward_mean,
            "Backward Std (s)": backward_std
        })
        
        print(f"  Forward: {forward_mean:.4f}s ± {forward_std:.4f}s")
        print(f"  Backward: {backward_mean:.4f}s ± {backward_std:.4f}s")
    
    # 创建DataFrame并返回
    results_df = pd.DataFrame(results)
    print("\n结果汇总：")
    print(results_df)
    
    # 保存结果到csv文件
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/exp1_timing_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    exp1()