# Task 4.A —— 使用独立测试集和正运动学评估 ANN

依据 `Coursework_1_2627.pdf` 第 12 页。在当前项目根目录运行；仅需 Python 3 和 NumPy，不需要启动 Webots。

## 运行

```console
python scripts/evaluate_ann.py
python scripts/verify_task4a.py
```

首次实现时已运行上述评估和复核，默认结果位于 `results/task4a/`。如果数据和模型没有变化，无需重新运行。重复运行会覆盖该目录内同名结果。也可以指定其他输出目录：

```console
python scripts/evaluate_ann.py --output results/task4a_run2
python scripts/verify_task4a.py --results results/task4a_run2
```

## 输入和评估过程

加载 Task 3 已选定的三个模型：`models/ik_shared_1000.npz`、`models/ik_shared_5000.npz` 和 `models/ik_shared_20000.npz`。每个模型分别使用 `datasets/ned_test.csv` 和 `datasets/scara_test.csv` 的全部 500 个样本，共得到 6 组评估、3000 条预测记录。

1. 从测试集读取目标 TCP 坐标，坐标系为机械臂自身的 ARM 坐标系，单位为米。
2. 将机械臂标识和目标坐标输入模型，复用训练时保存的坐标标准化参数。
3. 使用 `robot_config.py` 中的共享规则，将三个归一化输出还原为物理关节值。
4. 将预测关节值传入 Task 1 的 `fk_ned.fk()` 或 `fk_scara.fk()`，得到预测 TCP 坐标。
5. 逐样本计算位置误差和关节误差，再分别统计均值、标准差。

评估不会训练网络、重新拟合标准化参数或根据测试结果选模型。三个模型全部评估。沿用 Task 3 输出策略：不裁剪、不修正肘关节分支、不使用解析逆运动学回退。非正肘角等预测也保留在统计中，并单独计数。

程序校验模型与 Task 3 选模记录的哈希、训练源代码、数据配置与 FK 源代码哈希。还会校验测试集的行数、哈希、标签范围、正肘角、唯一性、参考关节值与 FK 坐标的一致性，并确认测试集的位置及关节标签与训练集、验证集不重叠。读取训练集和验证集仅用于完整性及重叠检查。

## 严格采用 PDF 的误差定义

### 位置误差

`e_position = sqrt((x_pred-x_true)^2 + (y_pred-y_true)^2 + (z_pred-z_true)^2)`

单位为米。汇总 CSV 额外提供毫米列，便于查看。

### 旋转关节误差

`delta_theta = atan2(sin(theta_true-theta_pred), cos(theta_true-theta_pred))`

这是**带符号的环绕角差**，单位为弧度（rad）。按 PDF 公式统计，不额外取绝对值或平方。正负误差可能在均值中相互抵消，因此均值和标准差需要同时保留。

### 直线关节误差

`e_d = (d_true-d_pred)^2`

这是 PDF 指定的**平方误差**，单位为 **m²**。仅适用于 SCARA 的第三个输出 `shaft_linear_motor`。其均值不是 RMSE，也不是以米为单位的平均绝对误差；其标准差是逐样本平方误差的标准差，单位同样是 m²。

### 标准差约定

所有统计采用总体标准差 `ddof=0`，即以本次完整的 500 个测试样本为统计总体。此约定保存在 `summary.json` 中。

| 输出 | Ned 电机 | Ned 误差单位 | SCARA 电机 | SCARA 误差单位 |
| --- | --- | --- | --- | --- |
| 1 | joint_1 | rad | base_arm_motor | rad |
| 2 | joint_2 | rad | arm_motor | rad |
| 3 | joint_3 | rad | shaft_linear_motor | m² |

## 输出文件

- `results/task4a/position_summary.csv`：6 行，3 个模型 × 2 种机械臂的位置误差均值及标准差，包含米、毫米两套显示单位。
- `results/task4a/joint_summary.csv`：18 行，3 个模型 × 2 种机械臂 × 3 个关节，记录电机名称、误差定义、单位、均值及标准差。
- `results/task4a/{small,medium,large}_{ned,scara}_samples.csv`：6 个逐样本文件，每个 500 行。包含目标及预测 TCP、真实及预测归一化关节值、真实及预测物理关节值、四项误差、非正肘角及物理限位标记。
- `results/task4a/summary.json`：完整统计及可复核信息，包括模型和测试数据哈希、源代码哈希、软件版本、样本数、输出策略和标准差约定。
- `results/task4a/verification.json`：独立复算检查的结果。

逐样本输出沿用原始测试集顺序，`sample_index` 从 1 开始。没有重新抽样或打乱；后续 Task 4.B 可以使用原测试集的前 100 行与之对应。

## 验证

```console
python -m unittest discover -s tests -v
```

新增测试覆盖跨越 ±π 时的角差符号、直线关节平方误差、总体标准差、两种机械臂的理想零误差，以及使用预测关节值计算 FK、保留非正肘角预测的行为。

`verify_task4a.py` 使用标准库 `math` 和 `statistics`，从逐样本 CSV 独立复算 FK、每项误差、均值及标准差，核对原测试集顺序、物理值还原、异常标记、6 组位置统计和 18 组关节统计，并对照 JSON 与汇总 CSV。它不调用评估程序的误差或统计函数。

Task 4.A 输出属于实际程序运行得到的技术记录；海报、视频中的最终解释和反思仍需由小组自行完成。Task 4.B 的 Webots 实测不包含在此脚本中。
