# Secret Sauce — 清洁室 GPU 核素移植流水线

**目标：** 从闭源 CUDA 核心中提取行为规范，并自动生成开源、跨平台（Triton）兼容的等价实现——字节级匹配。

**状态：** end-to-end 流水线已于三个不同核型中验证通过，包括一个**真实 cuBLAS 生产核**。
`port.py <cubin> <kernel_cu> [n]` 从单个 cubin 即可运行 spec 提取 → 生成 → 验证。

> 本文档是 `README.md` 的**中文节选**。外部 cubin 解析器、cudad 反编译、ZLUDA、
> sglang 生产核细节等章节目前仅存在于英文版。

## 验证核

| 核 | 源代码 | 形状 | 冗余度 | 大小 | 匹配 |
|---|---|---|---|---|---|
| `custom` | 手写 | 每个线程块减少 | 是（×32） | 256–16384 | YES |
| `vec_add` | 手写 | 逐元素 | 否 | 1024 | YES（全数组） |
| `transpose` | **cuBLAS fatbin** | 分区转置 | 否 | 16–128 | YES |

转置核是从 `cublas_fatbin.2.sm_80.cubin` 提取的 cuBLAS fatbin 转置核（64KB，真实生产 SASS）。流水线仅从 SASS 本身就检测到了其功能集，并生成了一个匹配的 Triton 核。

## 文件

| 文件 | 用途 |
|---|---|
| `kernel.cu` | "闭源"目标——共享内存分区、`__syncthreads()`、 warp shuffle 减少 |
| `vec_add_kernel.cu` | 参考逐元素核（无共享内存，无 shuffle） |
| `transpose_kernel.cu` | 参考分区转置核（cuBLAS 风格银行冲突填充） |
| `port.py` | **端到端驱动。** `port.py <cubin> <kernel_cu> [n]` |
| `verify_transpose.py` | 2D 转置验证器：`verify_transpose.py <cubin> <kernel_cu> <rows> <cols>` |
| `spec_extractor.py` | 从 cubin 提取行为规范（SASS + PTX + rea）→ `spec.json` |
| `generate_triton.py` | 读取 `spec.json`，输出干净室 Triton 核 → `generated_kernel.py` |
| `generated_kernel.py` | 自动生成的 Triton 核（`generate_triton.py` 的输出） |
| `custom.cubin` / `.ptx` | 编译的目标核（sm_80） |
| `vec_add.cubin` / `.ptx` | 编译的参考核 |
| `cublas_fatbin.2.sm_80.cubin` | 提取的 cuBLAS 转置核（sm_80） |
| `sass_emu.py` | **可执行的 SASS 模拟器**——运行 cubin，产出其输出缓冲区 |
| `sass_verify.py` | 同一 cubin 分别在 GPU 与模拟器上运行，逐字节比对输出 |
| `sass_opcheck.py` | 针对单条 SASS 指令语义的差分测试（GPU vs 模拟器） |
| `Decompile.java` | Ghidra 无头后处理脚本 |
| `cublas_fatbin.bin` | cuBLAS 的生产级 GPU fatbin——sm_75/80/86/90/100/120 共 1,261 个核 |
| `cublas_gpu_evidence.json` | REA 关于那个 fatbin 的库存 |

## 流水线

```
1. EXTRACT SPEC
   spec_extractor.py <cubin> spec.json
   → kernel name, SM target, shared_bytes, feature set, register count, SASS, PTX

2. CLEAN-ROOM GENERATE
   generate_triton.py spec.json generated_kernel.py
   → Triton kernel composed from feature templates. Never sees CUDA source.

3. VERIFY
   port.py <cubin> <kernel_cu> [n]
   → compiles the .cu → ctypes, runs generated Triton on SAME input tensors,
     asserts torch.equal(out_cuda, out_triton)
```

## 关键行为发现

第一个生成的核失败了——而这正是问题的所在。

该 CUDA 核在每一个 lane 中重复计算完整的点积（FFMA 遍历所有 256 个元素），然后 lane 0 写入结果 warp 中的 warp shuffle 链（偏移量 16/8/4/2/1）将 warp 内所有值相加 → 结果 = 点积 × 32。

一个朴素的端口计算了数学上正确的单个点积（77192.5），然后与 CUDA 输出静默偏离（2470160.0）。偏差比例精确为 32.0。

**编译器完全移除了 warp shuffle**——因为所有 lane 都持有相同的值，`SHFL` 指令在 SASS 中完全不出现。只剩下 `BSYNC` / `BAR.SYNC` 和长长的 FFMA 链。所以提取器不能仅仅通过搜索 `SHFL` 来检测它；而是要检测冗余签名：FFMA 计数 ≥ 32 + `bar_sync` 特性存在。

**教训：** 干净室端口必须复现**行为**，而不仅仅是"改进"内核。
提取器标记了 `full_reduce_redundancy`，生成器输出 `full * 32.0`——这不是一个"更好的"核，而是一个**忠实**的核。

## 工具链配置（经过重启仍然有效）

```bash
# Python 环境（持久化——.venv 在项目目录中，而不是 /tmp）
python3 -m venv .venv
.venv/bin/pip install torch triton numpy --index-url https://download.pytorch.org/whl/cu128
# 使用 gcc-15 作为宿主编译器（gcc 16 会破坏 torch 2.11 头文件）
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15
```

## CUDA 堆栈架构（已验证）

```
app → libcudart.so → dlopen libcuda.so.1 (驱动)
app → libcublas.so → libcublasLt.so → dlopen libcuda.so.1
GPU: 1,261 fatbin kernels → SASS on sm_75..sm_120
```

每个 CUDA 库都是 dlopen/dlsym 包装——没有硬链接 libcuda.so.1。
真正的 IP（kernel-mode 驱动、MLIR 前端、优化过的 fatbin 核）**不在** `cuda_ref/`（symlinked `/opt/cuda/lib64/`）中。

## 可执行的 SASS 模拟器，已于 GPU 上逐字节验证

`sasssim.py` 原本试图用符号化方式追踪 scatter 索引，但卡在循环摘要（loop
summarization）上。`sass_emu.py` 换了一条路：**直接执行 SASS**。它像硬件那样运行
cubin——32 lane 的 warp、每 lane 独立的寄存器与谓词、真实可字节寻址的
global/shared/constant 内存、跨 warp 的屏障——最终产出核的真实输出缓冲区，而不是
一条追踪记录。

没有针对单个核的查表，不读取源码，也没有任何修补。未实现的指令会**直接报错**而不是
近似估算，因此错误答案无法蒙混过关。

| 文件 | 作用 |
|---|---|
| `sass_emu.py` | 模拟器。输入 `cuobjdump -sass` 文本，输出内存状态 |
| `sass_verify.py` | 在 GPU 上加载**同一个 cubin**，逐字节比对输出 |
| `sass_opcheck.py` | 针对单条 SASS 指令的差分测试（GPU vs 模拟器） |

```bash
.venv/bin/python sass_verify.py            # 全部核
.venv/bin/python sass_verify.py --sweep    # 覆盖不同尺寸/形状的 18 个用例
.venv/bin/python sass_opcheck.py           # 逐指令差分测试
```

### 验证结果（RTX 4080，sm_89，CUDA 13.4）

每一行都是**整个输出缓冲区**的逐字节匹配；同时用独立的 numpy 参考实现确认 GPU
一侧的结果确实正确（测试输出中的 `ref==gpu`）——因此不可能出现“双方一起算错却判定
通过”的情况。

| 核 | 覆盖的形状 | 用例 | 结果 |
|---|---|---|---|
| `vec_add` | 逐元素、谓词退出 | n = 256, 1024, 4096 | 字节相等 |
| `custom` | 共享内存、`BAR.SYNC`、256 次 FFMA 链、×32 冗余 | n = 512, 1024, 2048 | 字节相等 |
| `transpose` | 二维 grid/block、带填充的共享分区、银行冲突布局 | 64×48 | 字节相等 |
| `moe_permute_prepare` | 纯整数、循环、`lower_bound` 二分查找、**scatter** | 6 种配置，int64 开/关 | 字节相等 |

`moe_permute_prepare` 就是上一节曾被称为“前沿”的 sglang 生产核。它的两个输出都
匹配：`expert_offsets`（17 个值）以及被 scatter 写入的 `src2dst`（100 个值，索引由
数据决定）。

scatter 不再是黑盒。`0xae0` 处的 store 用 `LEA R6, P1, R10, c[0x0][0x178], 0x2`
加 `LEA.HI.X R7, R10, c[0x0][0x17c], R11, 0x2, P1` 计算地址——这是一个 64 位寄存器
对，基址是 `src2dst` 指针，索引是 `reorder_ids[i]`。执行这对指令，在
`reorder_ids[i] = 5` 时得到 `0x10003014` = `src2dst + 5*4`，与 GPU 实际写入完全一致。

### 做对这些事情付出了什么

以下每一条都是通过与硬件的差分测试发现的，**不是**靠读助记符推断——每一条在最初都
先给出了“看起来合理但实际错误”的结果：

- `SHF d, a, shift, b`——操作数顺序为 `dst, a, shift, b`，拼接的漏斗是 `{b : a}`
  （`b` 是高半部分）。符号提取惯用法依赖这一点。
- `SEL d, a, b, P` —— `P ? a : b`；**第一个**源操作数是条件为真的分支。
- `IMAD.IADD` / `IMAD.SHL` —— 并不是另一种运算；`.IADD` 只是标注乘数为 1，
  两者都归约为通用的 `a*b + c`。
- `LOP3` 的真值表索引是 `(a<<2)|(b<<1)|c`，真正的源操作数紧邻 LUT 字节之前
  （某些形式下前面还有填充操作数）。
- `ISETP.<cmp>.<combine> P, PT, a, b, P2`——第 1 个操作数是**门控**（通常是 `PT`），
  第 4 个才是参与合并的值。把门控误读为 OR 输入会导致所有 lane 提前退出。
- `[R7.X4]` 是 32 位带缩放的地址；只有 `[R2.64]` 才是寄存器对。把带缩放基址
  当成寄存器对读取，会把相邻寄存器并入地址。
- 核参数在常量库中按自然边界对齐——`int a; int* p` 中 `p` 位于
  `c[0x0][0x168]`，而不是 `0x164`。
- `LDG`/`LDS` 是目标在前，`STG`/`STS` 是地址在前。
- 立即数 `0xffffffe` 是正数 28 位值（`+268435454`），不是 `-2`。

### 模拟器不建模的部分

以下均为诚实边界，全部通过**报错**而非猜测来实现：

1. **`LEA.HI` 的带符号除法修正（移位大于 1 时）。** ptxas 会把带符号的
   `a / 3`、`/4`、`/7` 转换成魔数乘法加 `LEA.HI`，而我未能推导出负数操作数下的
   移位规则：没有任何公式能同时匹配 `a = -1, -7, -19, -1023` 与硬件的答案。
   而移位为 1 的折半形式（即 `lower_bound` 的 `mid = lo + (hi-lo)/2`）**已验证且精确**。
   不含除法的核——上面四个都是——不受影响。这是已知的唯一一处指令级缺口，
   会以 `Unsupported` 具名报错。
2. **需要逐位精确超越函数结果的浮点运算。** `FADD`、`FFMA`、`FMUL`、`HFMA2`、
   `FMNMX`、`I2F`、`F2I` 通过 Python float 舍入到 fp32，在本文所有核上都逐字节一致，
   但 `MUFU`（倒数、exp2、log2）是用双精度计算的。它在 MoE 核（用 `MUFU.RCP`
   实现无符号除法）上匹配，但不保证普遍成立。
3. **完全未覆盖：** 简单模式之外的 `SHFL`、`CALL`/`RET`（没有子程序栈）、
   源码中列出的 add/max/min/exch 之外的 `ATOM`/`RED` 变体、`TEX`/`SURFACE`，
   以及异步的 `cp.async`/TMA 系列。`sm_90` 及以上的指令不在范围内；
   这里全部是 `sm_80` SASS。
4. **性能。** 它是指令解释器：n=2048 时的 `custom` 约 2.6 万条指令，MoE 核约 1.5 千条。
   用于验证足够，但不能替代在 GPU 上实际运行。

`sasssim.py` 保持原样：它记录了符号化追踪的尝试以及循环摘要为何困难。
就“真正复现输出”而言，`sass_emu.py` 取代了它。
