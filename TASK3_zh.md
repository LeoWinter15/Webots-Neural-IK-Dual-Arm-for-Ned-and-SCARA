# Task 3 —— 两种机械臂共用的人工神经网络逆运动学

本实现依据 `Coursework_1_2627.pdf` 第 7、10–11 页的要求。请在本项目根目录运行下列命令。仅依赖 Python 3 和 NumPy，训练时不需要 Webots。沿用 Task 1、Task 2 已有的控制器、正运动学（FK）实现、数据集和关节归一化规则。

## 训练三个模型

```console
python scripts/train_ann.py
```

| 模型 | Task 2 中每种机械臂划分前的样本数 | 合并后的训练样本数 | 合并后的验证样本数 | 输出文件 |
| --- | ---: | ---: | ---: | --- |
| small（小型） | 1000 | 1600 | 400 | models/ik_shared_1000.npz |
| medium（中型） | 5000 | 8000 | 2000 | models/ik_shared_5000.npz |
| large（大型） | 20000 | 32000 | 8000 | models/ik_shared_20000.npz |

每个模型都是一个同时使用 Ned 和 SCARA 数据训练的共享网络。small、medium、large 指训练数据集的规模，三个模型的网络宽度相同。训练程序读取 12 个指定的训练集、验证集 CSV 文件以及 `metadata.json`，不会打开测试集 CSV，也不会将测试集用于预处理参数拟合、参数选择或评估。

训练前，程序会检查文件哈希、样本行数、数值是否有限、归一化配置、肘关节标签是否为正、数据集的嵌套关系，以及训练集与验证集的位置是否重叠。测试集与其他集合不重叠的保证来自 Task 2 的生成过程和数据来源记录；Task 3 不会重新打开测试集进行检查。

## 网络结构与训练设置

- 输入：`[is_scara, is_ned, x, y, z]`。Ned 的标识为 `[0, 1]`，SCARA 为 `[1, 0]`。坐标以米为单位，位于各机械臂自身的 ARM 坐标系中，不是 Webots 世界坐标。
- 坐标标准化：仅使用合并训练集的 xyz 计算均值和总体标准差。验证和预测时复用保存的标准化参数。机械臂标识始终保持为 0 或 1。
- 网络结构：`5 -> 64 -> 64 -> 3`，每个隐藏层和输出层均使用 `tanh`，共 4739 个参数。权重采用 Xavier 均匀分布初始化，偏置初始化为零，计算使用 `float64`。
- 输出：三个归一化关节值，归一化采用 `scripts/robot_config.py` 中的物理关节限位。输出层的 `tanh` 将数值限制在 `[-1, 1]`。
- 损失函数：对所有样本、全部三个归一化输出计算均方误差（MSE）。两种机械臂的样本数相同，因此在选模指标中权重相同。
- 优化器：Adam，`beta1=0.9`、`beta2=0.999`、`epsilon=1e-8`，小批量大小为 256。
- 候选学习率：0.001 和 0.003。两组实验均使用相同的初始化种子 42，以及相同的确定性小批量打乱顺序（打乱种子为 43）。
- 每组候选参数最多训练 500 轮。早停依据独立的合并验证集，耐心值为 60 轮，绝对改善阈值为 `1e-7`。始终保留验证集 MSE 最低轮次的参数，即使某次改善小于早停阈值，也会保存该最佳参数。
- 对每种数据规模，保存验证集 MSE 最低的候选模型。由于验证集参与选模，这些数值不能视为无偏的最终性能估计；独立测试集上的最终评估属于 Task 4。

可以通过命令行调整设置，例如：

```console
python scripts/train_ann.py --hidden 64 64 --learning-rates 0.001 0.003 --epochs 500 --batch-size 256 --patience 60 --seed 42 --models models_run2 --results results/task3_run2
```

使用默认路径重新运行时，会覆盖三个默认模型文件和当前训练历史文件。如需保留另一组实验，请为 `--models` 和 `--results` 指定不同目录。

在相同数值计算环境中，随机种子可以用于复现训练；不同版本的 NumPy 或 BLAS 可能造成浮点计算差异。模型记录了 Python、NumPy 版本，以及源代码和数据文件的 SHA-256 哈希。

## 保存的记录与模型格式

- `models/ik_shared_N.npz`：数值形式的权重、偏置、坐标标准化参数，以及内嵌的 JSON 元数据。加载时使用 `allow_pickle=False`，不依赖 sklearn 或 pickle。
- `models/ik_shared_N.json`：便于阅读的元数据副本，包含完整机械臂配置、超参数、选中的学习率与训练轮次、两组候选实验、验证指标，以及源代码和数据哈希。
- `results/task3/*_candidate_*_history.csv`：逐轮记录训练集 MSE、合并验证集 MSE，以及 Ned、SCARA 各自的验证集 MSE。
- `results/task3/summary.json`：选中的模型、哈希、耗时和全部候选实验记录。

每种机械臂的验证指标包括归一化 MSE、每个输出的平均绝对误差，以及反归一化后肘关节角度非正的预测数量。

加载模型时，会核对当前机械臂配置、输入输出顺序、各层参数形状，以及参数是否为有限数值。训练程序会重新加载每个选中的模型，并检查它在两种机械臂验证输入上的预测是否与保存前完全一致。

## 预测电机目标值

```console
python scripts/predict_ann.py ned 0.09734 0.16239 0.15728
python scripts/predict_ann.py scara 0.53558 0.27547 0.082 --model models/ik_shared_5000.npz
```

这些示例坐标来自手动指定姿态的 FK 结果，并进行了四舍五入：Ned 关节值为 `-0.54、0.4、0.9 rad`；SCARA 关节值为 `0.3 rad、0.5 rad、-0.1 m`。它们不是独立测试集中的评估样本。

默认加载大型模型。命令行输出归一化预测值，以及带真实电机名称和单位的物理关节值。后续位于 `scripts/` 下的 Task 4 脚本可以通过以下接口调用模型：

```python
from pathlib import Path
from ann_model import IKModel

root = Path(__file__).resolve().parent.parent
model = IKModel(root / "models" / "ik_shared_20000.npz")
xyz = [[0.53558, 0.27547, 0.082]]  # N × 3 数组，ARM 坐标系，单位为米
q_norm = model.predict_normalised("scara", xyz)
q_motor = model.predict_joints("scara", xyz)
```

| 输出 | Ned 电机 / 单位 | SCARA 电机 / 单位 |
| --- | --- | --- |
| 1 | joint_1 / rad（弧度） | base_arm_motor / rad（弧度） |
| 2 | joint_2 / rad（弧度） | arm_motor / rad（弧度） |
| 3 | joint_3 / rad（弧度） | shaft_linear_motor / m（米） |

物理关节值通过已有共享配置中的 `denormalise_joints` 获得。预测后不进行数值裁剪、肘关节分支修正、解析逆运动学回退或工作空间投影。

`tanh` 可以保证输出对应物理关节限位，但不能在数学上保证肘关节角度为正：训练标签只覆盖较宽物理关节范围中的正角度分支。命令行会提示非正肘角，验证记录会统计其出现次数。对于不可达目标或训练分布以外的目标，预测结果不保证有效。Task 4 应继续采用一致的输出处理策略；如果修改策略，需要明确记录。

## 验证方法

```console
python -m unittest discover -s tests -v
python scripts/verify_task3.py
```

检查内容包括：使用中心有限差分核对小型网络中每个参数的梯度；验证 Adam 首次更新及训练损失下降；核对模型保存、重新加载后的预测；检查两种机械臂的物理值换算、配置不匹配时的拒绝行为、机械臂标识、输出范围与输入合法性；检查训练集和验证集的数据来源；确认训练读取接口拒绝测试集类型。这些测试仅使用合成数组和已有的训练集、验证集文件。

训练结束后，`verify_task3.py` 还会检查三个模型的哈希、源代码哈希、元数据副本，以及所选候选模型和训练轮次是否与损失历史一致，并使用保存的模型重新计算验证集 MSE。结果写入 `results/task3/verification.json`。该脚本不会评估 Task 4 的测试集。

当前机器上的 scikit-learn 因与 NumPy 存在兼容性问题而无法导入。Task 3 不导入 scikit-learn，因此无需为本实现更改环境。正式提交前，仍需完成在 GRID 或另一台计算机上运行，以及基于 Webots 的验证等项目级检查。

## 本次实现范围

本次新增 Task 3 的训练与预测功能。Task 4 的测试集、FK、Webots 评估，以及 Task 5 的提交材料不在本次实现范围内。训练历史属于技术记录，不是海报或视频的结论，也不是小组的 AI 反思声明。

请保留本次聊天，作为 AI 辅助代码生成与自动化检查的记录；小组仍需自行审查并理解最终提交的实现。
