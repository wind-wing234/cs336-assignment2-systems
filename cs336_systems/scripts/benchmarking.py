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
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW, get_cosine_lr

model_sizes = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12}, # 要3G左右的显存
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25}, # 要40G左右的显存
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32} # 要60G左右的显存
}


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

def run_LM_with_optimizer(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    context_length: int = 256,
    batch_size: int = 4,
    use_autocast: bool = False,
) -> tuple[Callable, Callable, Callable]:
    """
    返回"运行一次前向传播(+反向传播+优化器步骤)"的函数
    
    Args:
        d_model: 模型维度
        d_ff: 前馈网络维度
        num_layers: 层数
        num_heads: 注意力头数
        context_length: 上下文长度
        batch_size: 批量大小
        use_autocast: 是否使用混合精度训练
    """
    # 默认参数
    vocab_size = 10000
    rope_theta = 10000.0
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

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )

    # 如果使用autocast，初始化梯度缩放器
    scaler = torch.amp.GradScaler("cuda") if use_autocast else None

    # 获取输入batch
    dataset = np.load(dataset_path, mmap_mode='r+')
    inputs, targets = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )

    logits: torch.Tensor = None
    loss: torch.Tensor = None
    
    def forward():
        nonlocal logits
        if use_autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                logits = model(inputs)
        else:
            logits = model(inputs)

    def backward():
        nonlocal loss
        if use_autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                loss = cross_entropy(logits, targets)
            scaler.scale(loss).backward()
        else:
            loss = cross_entropy(logits, targets)
            loss.backward()
    
    def optimizer_step():
        if use_autocast and torch.cuda.is_available():
            scaler.unscale_(optimizer)
            clip_gradient(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_gradient(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    return forward, backward, optimizer_step


def exp1():
    """
    对不同大小的模型进行基准测试，测量前向传递和前向+后向传递的时间
    """
    
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

def exp2():
    """
    在small观察未预热、预热1次、2次、5次的前向、反向传递时间
    """
    num_warmups = [0, 1, 2, 5]
    num_steps = 10
    results = []
    model_size = "small"
    for warmup in num_warmups:
        print(f"Testing with {warmup} warmup(s)...")
        forward, backward = run_LM(
            d_model=model_sizes[model_size]["d_model"],
            d_ff=model_sizes[model_size]["d_ff"],
            num_layers=model_sizes[model_size]["num_layers"],
            num_heads=model_sizes[model_size]["num_heads"]
        )

        runs = [forward, backward]
        time_results = benchmark(
            description=f"{model_size} model forward and backward with {warmup} warmup(s)",
            runs=runs,
            num_warmups=warmup,
            num_trials=num_steps
        )

        forward_times = time_results[forward.__name__]
        backward_times = time_results[backward.__name__]

        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)
        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)

        results.append({
            "Warmups": warmup,
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
    results_df.to_csv(f"results/exp2_{model_size}_timing_results.csv", index=False)
    return results_df

def exp3(model_size="2.7B", context_lengths=[128, 256, 512], profile_full_training=False, use_autocast=False):
    """
    对指定大小的模型进行内存分析，测量不同上下文长度下的内存使用情况
    
    Args:
        model_size: 模型大小名称，默认为"2.7B"
        context_lengths: 要测试的上下文长度列表
        profile_full_training: 如果为True，分析完整训练步骤；如果为False，仅分析前向传递
        use_autocast: 是否使用混合精度训练
    """
    
    print(f"内存分析: {model_size} model {'(完整训练步骤)' if profile_full_training else '(仅前向传递)'} {'使用混合精度' if use_autocast else '使用全精度'}")
    
    # 确保结果目录存在
    os.makedirs("results/memory_profiles", exist_ok=True)
    
    # 获取模型参数
    model_params = model_sizes[model_size]
    
    for context_length in context_lengths:
        print(f"分析上下文长度: {context_length}")
        
        # 创建运行函数
        forward, backward, optimizer_step = run_LM_with_optimizer(
            d_model=model_params["d_model"],
            d_ff=model_params["d_ff"],
            num_layers=model_params["num_layers"],
            num_heads=model_params["num_heads"],
            context_length=context_length,
            use_autocast=use_autocast
        )
        
        # 预热阶段
        for _ in range(5):  # 进行5次预热
            forward()
            if profile_full_training:
                backward()
                optimizer_step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # 开始记录内存历史
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        
        for _ in range(5):  # 进行5次实际分析
        # 执行要分析的操作
            forward()
            if profile_full_training:
                backward()
                optimizer_step()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 保存内存快照
        profile_type = "full_training" if profile_full_training else "inference"
        precision_type = "mixed" if use_autocast else "full"
        output_file = f"results/memory_profiles/{model_size}_ctx{context_length}_{profile_type}_{precision_type}.pickle"
        torch.cuda.memory._dump_snapshot(output_file)
        
        # 停止记录历史
        torch.cuda.memory._record_memory_history(enabled=None)
        
        print(f"已保存内存快照到 {output_file}")
        torch.cuda.empty_cache()
        
    print("内存分析完成！")
    print("请使用PyTorch的内存可视化工具查看结果：https://pytorch.org/memory_viz")
    print("命令行方式：python -m torch.cuda.memory_viz results/memory_profiles/[snapshot_file]")

if __name__ == "__main__":
    # exp2()
    
    # 仅前向传递(推理)的内存分析 - 全精度
    exp3(profile_full_training=False, use_autocast=False)
    
    # 仅前向传递(推理)的内存分析 - 混合精度
    exp3(profile_full_training=False, use_autocast=True)
    
    # 完整训练步骤的内存分析 - 全精度
    exp3(profile_full_training=True, use_autocast=False)
    
    # 完整训练步骤的内存分析 - 混合精度
    exp3(profile_full_training=True, use_autocast=True)