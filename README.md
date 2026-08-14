# ComfyUI H3 Transfer Boost

面向 MiniMax H3 大模型低显存推理的 ComfyUI 自定义节点。目标不是跳步，而是减少或隐藏 CPU RAM → GPU VRAM 权重交换成本。

当前版本提供可复现的判断工具与立即可用的交换调优：

- **H3 Weight Compressibility Analyzer**：扫描模型中的 1-byte 权重（INT8/FP8），按张量估算 ANS 可压缩率，先判断你的 H3 checkpoint 是否值得做压缩交换。
- **H3 PCIe Transfer Benchmark**：测量 pinned-memory H2D 带宽；安装 nvCOMP 后，额外真实测量“ANS 压缩数据 H2D + GPU 解压”端到端有效带宽，并逐字节验证无损。
- **H3 Async Offload Tuner**：只在该 MODEL 执行期间调整 ComfyUI 异步卸载流数量，保留已有 model wrapper，适合测试 2/3/4 流是否能更好地覆盖传输。

> 当前版本没有声称实现透明的“整模型压缩卸载”。在 ComfyUI 最新 DynamicVRAM/VBAR 中，权重可能来自 gathered pinned buffer；绕过它直接替换 Parameter 会破坏 patch/LoRA/量化布局语义。先用本项目测出压缩交换的真实收益，再进入 VBAR 层集成，是更可靠的路线。

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
