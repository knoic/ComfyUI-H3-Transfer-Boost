# ComfyUI H3 Transfer Boost

面向 MiniMax H3 大模型低显存推理的 ComfyUI 自定义节点。目标不是跳步，而是减少或隐藏 CPU RAM → GPU VRAM 权重交换成本。

当前版本提供可复现的判断工具与立即可用的交换调优：

- **H3 Weight Compressibility Analyzer**：扫描 INT8、FP8、BF16 和 FP16 权重的原始字节熵，按张量估算 ANS 可压缩率，先判断 checkpoint 是否值得做压缩交换。
- **H3 PCIe Transfer Benchmark**：测量 pinned-memory H2D 带宽；安装 nvCOMP 后，额外真实测量“ANS 压缩数据 H2D + GPU 解压”端到端有效带宽，并逐字节验证无损。
- **H3 Async Offload Tuner**：只在该 MODEL 执行期间调整 ComfyUI 异步卸载流数量，保留已有 model wrapper，适合测试 2/3/4 流是否能更好地覆盖传输。
- **H3 Compressed Swap (Experimental)**：截获 DynamicVRAM/VBAR 的 gathered pinned 权重传输，首遍建立 ANS 缓存，后续改走 compressed H2D + GPU decode。

Analyzer 和 Benchmark 是输出节点，执行后 JSON 会直接显示在节点上；也可以把 STRING 输出接到最新版 ComfyUI 自带的 `Preview as Text`。

## 实验性压缩交换

连接顺序：

```text
H3 Model Loader → LoRA / Model Patch → H3 Compressed Swap → Sampler
```

推荐起始参数：

```text
streams: 3
min_tensor_mb: 8
max_ratio: 0.80
cache_limit_gb: 8
fallback_on_error: true
```

- 首个采样步仍执行普通 H2D，并在同一 transfer stream 上压缩已整理好的 VBAR buffer，随后保存为 compressed pinned CPU cache。
- 后续采样步把 compressed blob 搬到每个 transfer stream 自己的 staging buffer，并由 GPU ANS 直接解压到原 VBAR 目标。
- `max_ratio=0.80` 表示实测压缩后必须不超过原大小的 80%；否则该权重永久回退普通传输。
- `cache_limit_gb` 限制额外的 compressed pinned RAM。当前实验版不会释放 ComfyUI 自己持有的原始 pinned 权重，因此会增加内存，但能减少后续 PCIe 传输量。
- 不要把它和 `H3 Async Offload Tuner` 串联；压缩节点已经包含独立的多流池。
- 每次模型调用后，ComfyUI 控制台会输出 `cached_tensors`、`measured_ratio`、`compressed_transfer_hits` 和 `errors`。
- 生成完成后，也可以把同一个压缩 MODEL 接到 `H3 Compressed Swap Stats`，再次 Queue 查看节点内 JSON 统计。

此路径当前仅支持 NVIDIA CUDA、nvCOMP、DynamicVRAM/VBAR 产生的单一 1D byte CPU source。其他传输、LoRA patch buffer、非 CUDA 后端和不可压缩权重保持 ComfyUI 原始路径。

BF16/FP16 也会进入同一无损 ANS 路径，因为 VBAR 的 gathered source 本身就是原始 byte buffer。浮点尾数往往提高熵，所以实际压缩率可能弱于 INT8；运行时先做熵筛选，再以 `max_ratio` 检查真实 nvCOMP 大小，不合格就自动使用原始 H2D。

要单独评估 BF16，在 `H3 PCIe Transfer Benchmark` 中选择 `bf16_like`。该模式生成 BF16 正态分布权重的真实字节布局，再测 compressed H2D + decode，而不是用 INT8 合成分布代替。

> 压缩交换目前是实验路径：它已接入 gathered pinned VBAR transfer source，但还没有让 ComfyUI 释放原始 pinned master。它优先验证推理交换速度与正确性，不应被当作降低总 RAM 的完成版。

## 安装

将仓库放到：

```text
ComfyUI/custom_nodes/ComfyUI-H3-Transfer-Boost
```

然后重启 ComfyUI。节点位于 `H3/Transfer Boost`。

nvCOMP 是可选依赖。根据 PyTorch 所带 CUDA 主版本二选一：

```bash
# CUDA 12
python -m pip install nvidia-nvcomp-cu12

# CUDA 13
python -m pip install nvidia-nvcomp-cu13
```

不安装 nvCOMP 时，分析器、普通 H2D 基准和异步流调优仍可使用。

## 推荐测试顺序

1. 对同一 H3 模型运行 `H3 Weight Compressibility Analyzer`。
2. 如果 `estimated_saved_percent` 小于约 15%，ANS 很可能不值得集成；如果大于 25%，继续。
3. 运行 `H3 PCIe Transfer Benchmark`，选择 `h3_like`，观察 `projected_transfer_speedup`。
4. 只有当该值明显大于 1.0 时，压缩交换才有硬件层面的收益空间。
5. 将 `H3 Async Offload Tuner` 接到采样器前，分别实测 2、3、4 流。以完整生成耗时为准，不以单个 kernel 时间为准。

## 为什么这条路线可能有效

普通交换每层需要搬运 `N` 字节。压缩交换的近似条件是：

```text
compressed_bytes / PCIe_bandwidth + compressed_bytes / GPU_decode_bandwidth
< uncompressed_bytes / PCIe_bandwidth
```

收益来自两点：PCIe 搬得更少；GPU 解压可在消费权重的计算流附近完成。代价是解压 kernel、临时缓冲和调度复杂度。FP8 或 GGUF 权重通常熵较高，可能几乎不可压；INT8 也必须实测，不能仅凭 dtype 判断。

## OneTrainer PR 带来的启发

- [#1644 Simplex offloading](https://github.com/Nerogar/OneTrainer/pull/1644)：CPU 保留 master，GPU 副本用完即丢弃，避免无意义 D2H 回写。推理权重只读，这一点尤其适用。
- [#1630 nvCOMP compressed weights](https://github.com/Nerogar/OneTrainer/pull/1630)：ANS 对部分 INT8 权重可显著减少传输量，但 FP8/GGUF 收益可能很低。
- [#1649 async offloading fixes](https://github.com/Nerogar/OneTrainer/pull/1649)：CUDA event、有限前瞻和 backpressure 是正确性要求，不只是性能细节。
- [#1680 fused linear epilogue](https://github.com/Nerogar/OneTrainer/pull/1680)：减少算子和激活占用可为预取环腾出显存，但不是交换压缩本身。

本仓库没有复制 OneTrainer 的底层实现；`h3_transfer_boost/nvcomp.py` 使用 NVIDIA nvCOMP 公开 Python API 独立实现。

## 下一阶段：透明压缩交换

基准确认有收益后，下一阶段应在 ComfyUI VBAR 的 transfer-source 层实现，而不是改 H3 block：

1. CPU 仅保留 pinned compressed master；GPU 不回写只读权重。
2. 为压缩 H2D 建立有界 ring buffer（建议 2–3 个槽位）。
3. transfer stream 搬压缩 blob，decode stream/GPU 解压到现有 VBAR cast buffer。
4. 用 CUDA event 建立 `copy → decode → matmul → recycle` 依赖和 backpressure。
5. cache key 必须包含权重签名、patch/LoRA 状态、dtype、shape 和 device；模型卸载或 patch 变化立即失效。
6. 为 batched ANS 去掉每层 device→host status readback，否则会在计算流制造约几十微秒级同步点。

这部分更适合向 ComfyUI 核心提交小型接口扩展；纯 custom node 猴子补丁会随 VBAR 内部变化失效。

## 限制

- 熵分析是压缩率估算，不等于实际 ANS 结果；实际结果看 nvCOMP benchmark。
- `h3_like` 是合成分布，不代表某个具体 checkpoint。
- Async Offload Tuner 调整的是进程级 ComfyUI 流池，但只在该模型调用期间改变启用数量；不建议并发执行多个不同配置的工作流。
- 增加流数会增加 staging/cast buffer 显存，显存已经贴边时反而可能 OOM。

## License

MIT。研究与设计参考见上方链接。
