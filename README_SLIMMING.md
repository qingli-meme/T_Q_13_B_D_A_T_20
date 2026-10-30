# QuRA V12 Method-Slimming Suite

Base repository state: commit `171b4b7` or a descendant containing V11/V12.

目标不是继续堆指标，而是回答：**V12 中哪些结构对 `ASR >= 99%` 且 `CA >= 91.3%` 真正必要？**

这个包不做参数 sweep。每个实验只删除或简化一个结构假设。

## 已经先做掉的等价瘦身

统一实现不再构造所有 source-class cores。方法只估计 target reference `c_t`，然后定义：

\[
\gamma_t=D_t(c_t),
\]

\[
u^\star(x)=\arg\min_u\|u-u_0(x)\|_2^2\quad\text{s.t.}\quad D_t(u)\ge\gamma_t.
\]

source class 不参与 endpoint 定义；类别仅用于日志诊断。

## 12 个顺序任务

| 顺序 | Variant | 问题 |
|---:|---|---|
| 00 | `reference_v12` | 在 target-core-only 清理后复现 V12，确认等价。 |
| 01 | `core_ray` | Dykstra 全局投影是否必要？闭式 source→target-core ray 第一次入域是否足够？ |
| 02 | `no_clean_anchor` | 显式 clean feature anchor 是否必要？ |
| 03 | `gamma_zero` | absolute endpoint 保持时，positive/deep target interior 是否必要？ |
| 04 | `no_augmentation` | dynamic crop/flip 是否必要？ |
| 05 | `all_samples` | 是否必须排除 pseudo-target calibration samples？ |
| 06 | `rank2` | rank-4 field 是否冗余？ |
| 07 | `rank1` | field 能否压到 rank 1？ |
| 08 | `late34_only` | 是否需要全网络 field，还是 layer3+4 足够？ |
| 09 | `layer4_only` | 能否只改 layer4？ |
| 10 | `core_ray_no_clean` | Core-Ray 成立后，clean anchor 能否同时删除？ |
| 11 | `core_ray_no_aug` | Core-Ray 成立后，augmentation 能否同时删除？ |

## Core-Ray 简化

令：

\[
u(\alpha)=u_0+\alpha(c_t-u_0),\quad\alpha\in[0,1].
\]

因为每个 normalized target margin 都是 affine：

\[
q_j(u(\alpha))=q_j(u_0)+\alpha[q_j(c_t)-q_j(u_0)].
\]

对每个违反 `q_j(u_0) < gamma_t` 的 boundary：

\[
\alpha_j=\frac{\gamma_t-q_j(u_0)}{q_j(c_t)-q_j(u_0)}.
\]

于是：

\[
\alpha^\star=\max_j\alpha_j,
\qquad
u^\star=u_0+\alpha^\star(c_t-u_0).
\]

它完全删除 Dykstra，仍保证 target depth floor，且 movement 不会超过 V6 target-core routing。它不声称是全局 Euclidean minimum，只是 ray 上最短可行点。

## 安装

把 ZIP 解压到仓库根目录。应看到：

```text
mqbench/structured_slimming_attack.py
ours/main/configs/slimming/
scripts/
install_slimming.py
```

执行：

```bash
python install_slimming.py
python scripts/verify_slimming_package.py
```

安装器只自动修改 `ours/main/main.py` 的 dispatcher，并在第一次修改时备份：

```text
ours/main/main.py.pre_slimming.bak
```

安装器可重复执行。

## 一次性用 tmux 挂全部实验

仓库根目录：

```bash
bash scripts/launch_slimming_tmux.sh
```

默认 session：

```text
qura_slimming
```

查看：

```bash
tmux attach -t qura_slimming
```

退出但不停止：`Ctrl-b d`。

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/launch_slimming_tmux.sh
```

指定 session 名：

```bash
bash scripts/launch_slimming_tmux.sh qura_slimming_gpu0
```

## 不使用 tmux

```bash
bash scripts/run_slimming_all.sh
```

总控脚本设计为：**某个实验失败也继续跑后面的实验**。

## 输出

每项 JSON：

```text
ours/main/output/slimming_<variant>_resnet18_cifar10.json
```

checkpoint：

```text
ours/main/model/resnet18+cifar10.slimming_<variant>_t0.pth
```

完整日志：

```text
results/logs/slimming_suite/
```

状态：

```text
results/slimming_suite/status.tsv
```

自动汇总：

```text
results/slimming_suite/summary.csv
results/slimming_suite/summary.md
```

## 回来后建议提交/给我的内容

```text
ours/main/output/slimming_*_resnet18_cifar10.json
results/logs/slimming_suite/
results/slimming_suite/status.tsv
results/slimming_suite/summary.csv
results/slimming_suite/summary.md
```

145 MB 左右的 checkpoint 不需要上传 GitHub。

## 判定规则

- `01_core_ray` 若接近 V12：主方法删掉 Dykstra。
- `02_no_clean_anchor` 若仍 `ASR>=99%` 且 CA 接近 reference：删 clean loss。
- `03_gamma_zero` 若 ASR 明显掉：positive depth 是必要机制。
- `04_no_augmentation` 若不掉：删 augmentation。
- `05_all_samples` 若不掉：删 source filtering。
- `06_rank2` / `07_rank1`：保留最小能维持 operating point 的 rank。
- `08_late34_only` / `09_layer4_only`：判断 whole-network rounding field 是否必要。
- `10` / `11` 用于验证单因素可删之后能否组合删除。

目标 operating point：

```text
硬条件：ASR >= 99%
Clean：  CA >= 91.3%
强结果：CA 91.6–92.0%+，ASR 99.3–99.8%+
```

最终目标不是得到最高分，而是找到**仍在 operating region 内的最小充分方法**。
