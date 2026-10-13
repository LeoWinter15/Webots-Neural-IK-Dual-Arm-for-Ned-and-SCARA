# 实验总结与视频演示操作指南

项目：Webots Neural IK for Ned and SCARA（webots-neural-ik-dual-arm）

本说明依据 Coursework_1_2627.pdf 和当前代码、实际结果整理，核对日期为 2026-10-09。用于理解实现和准备演示，不作为最终视频讲稿、幻灯片正文或结果反思。课程要求由小组自行撰写最终解释、设计理由、反思与结论。

## 1. 实验要求做什么

目标是：给定机械臂类型和期望末端位置 `(x,y,z)`，用人工神经网络预测三个活动关节的值，再驱动机械臂接近目标。只研究位置，不研究末端朝向。同一个已训练模型需要支持两种机械臂。

| 任务 | 要求 | 当前实现及证据 |
| --- | --- | --- |
| Task 1.A | 在 Webots 中输入关节值、驱动机械臂、稳定后读取 TCP | `ned_simple`、`scara_simple`、`tcp_pose` 控制器；`tcp_measurements.csv` |
| Task 1.B | 为两种机械臂实现正运动学 FK | `scripts/fk_ned.py`、`scripts/fk_scara.py` |
| Task 1.C | 将 FK 坐标与 Webots 坐标比较 | `scripts/compare_fk_webots.py`；`results/task1/` |
| Task 2 | 用随机关节值和 FK 生成三种规模、嵌套的训练/验证集，另建独立测试集 | `scripts/generate_datasets.py`、`datasets/metadata.json` 和 CSV |
| Task 3 | 每种数据规模将两臂数据合并，训练一个共享 ANN；用验证集调参 | `ann_model.py`、`train_ann.py`；三个 `.npz` 模型及训练日志 |
| Task 4.A | 三模型、两臂、完整测试集，统计 FK 位置误差和各关节误差 | `evaluate_ann.py`；3000 条预测、6 组位置统计、18 组关节统计 |
| Task 4.B | 每臂原测试集前 100 行，驱动 Webots 并统计真实仿真 TCP 误差 | `prepare_task4b.py`、`task4b_runner`、`summarize_task4b.py`；600 次成功实测 |
| Task 5 | 提交代码、A0 海报、视频、AI 反思声明等材料 | 代码和数值结果已具备；最终展示材料仍需整理 |

提交代码需要包含运行所需的数据、模型、世界及资源，并满足指定环境和允许库限制。小组还需在另一台电脑或 GRID 验证运行；每位成员独立完成 peer-marking。完成 Task 1–4 不等于完成全部提交材料。

## 2. 代码如何实现

### 2.1 关节与坐标

| 模型输出 | Ned | SCARA |
| --- | --- | --- |
| 第 1 个关节 | joint_1，rad | base_arm_motor，rad |
| 第 2 个关节 | joint_2，rad | arm_motor，rad |
| 第 3 个关节 | joint_3，rad | shaft_linear_motor，m |

Ned 的 joint_4、joint_5、joint_6 固定为零；SCARA 的 shaft_rotation_motor 固定为零。所有训练与评估位置均使用机械臂根节点的 ARM 坐标系，单位为米。

FK 完成“关节值 → TCP 坐标”。SCARA 使用平面两连杆旋转、几何偏移和伸缩轴位移；Ned 使用连杆实际偏移，先计算肩肘所在平面的距离和高度，再应用底座旋转。Ned 零位朝向 +Y，代码已计入这一约定。

`robot_config.py` 统一关节顺序、单位、物理限位和归一化。数据采样限制肘关节为正，但归一化仍使用完整物理限位：

```text
q_norm = 2 * (q - q_min) / (q_max - q_min) - 1
q = q_min + (q_norm + 1) * (q_max - q_min) / 2
```

### 2.2 数据集

随机生成关节值，用 FK 得到对应 xyz，然后将关节值归一化为监督学习标签。CSV 列为 `arm,x,y,z,q1_norm,q2_norm,q3_norm`。

| 每臂划分前样本数 | 每臂训练/验证 | 两臂合并训练/验证 |
| ---: | --- | --- |
| 1000 | 800 / 200 | 1600 / 400 |
| 5000 | 4000 / 1000 | 8000 / 2000 |
| 20000 | 16000 / 4000 | 32000 / 8000 |

较小训练集是较大训练集的前缀，验证集也采用同样的嵌套方式。每臂另有 500 行独立测试数据，训练和选模不读取测试集。Ned 和 SCARA 的生成种子分别为 42、43。

### 2.3 神经网络

```text
[is_scara, is_ned, x, y, z]
              ↓
     64 个 tanh 隐藏单元
              ↓
     64 个 tanh 隐藏单元
              ↓
     3 个 tanh 归一化输出
              ↓
 按机械臂类型反归一化为电机目标
```

网络为 `5 → 64 → 64 → 3`，共 4739 个参数，使用纯 NumPy 实现前向传播、反向传播和 Adam。Ned 标识为 `[0,1]`，SCARA 为 `[1,0]`。xyz 的均值和标准差只从训练集计算。

损失为归一化关节 MSE；批量 256；初始化种子 42；比较学习率 0.001、0.003；最多 500 轮；验证集早停耐心值 60 轮。每个候选保存验证 MSE 最低的参数，再在候选中选最优者。small/medium/large 是数据规模不同，网络结构相同，不是三种不同宽度的网络。

`predict_ann.py` 可以加载模型并预测任意指定 xyz 的关节目标。`tanh` 约束物理关节范围，但不能保证肘角一定为正，也不能保证任意输入坐标可达。预测没有使用解析 IK 回退。

### 2.4 两种评估方式

```text
Task 4.A：测试 xyz → ANN → 物理关节 → FK → 预测 xyz → 与目标比较
Task 4.B：测试 xyz → ANN → 物理关节 → Webots 电机 → Supervisor 实测 xyz → 与目标比较
```

4.B 的 ANN 预测在准备脚本中预先计算，仿真控制器执行保存的指令。因此它是“ANN 预测驱动的批量仿真”，并不是在 Webots 内每一帧重新运行网络。

4.B 同时检查活动及固定关节传感器，连续 20 步满足容差才记录为稳定。基本步长 16 ms，角度容差 0.002 rad，伸缩轴容差 0.0005 m，超时为 60 秒仿真时间。TCP 来自 `getPose(ARM)` 的仿真坐标。专用世界降低地面以给伸缩轴留出空间；ARM 坐标系未改变。

误差定义必须准确：

- 位置：三维欧氏距离，m；可乘 1000 显示为 mm。
- 旋转关节：`atan2(sin(q_true-q_pred), cos(q_true-q_pred))`，带符号，rad。
- 直线关节：`(d_true-d_pred)^2`，m²，不能标成 m 或 RMSE。
- 标准差均使用总体标准差 `ddof=0`。

## 3. 目前可展示的实测记录

2026-10-09 已重新运行模型复核、Task 4.A 独立复核、Task 4.B 汇总，全部通过。Task 4.B 六组均为 100/100、无缺失和当前失败记录。这里的“成功”是达到稳定判定，不表示 ANN 零误差或目标位置完全一致。

下表是程序现有输出的整理，单位为 mm，格式为均值 ± 标准差。

| 模型 | 4.A Ned，500 行 | 4.A SCARA，500 行 | 4.B Ned，100 行 | 4.B SCARA，100 行 |
| --- | --- | --- | --- | --- |
| small | 21.010 ± 17.448 | 16.773 ± 10.336 | 19.331 ± 13.751 | 15.341 ± 11.276 |
| medium | 11.259 ± 7.826 | 8.899 ± 7.871 | 10.493 ± 6.885 | 9.533 ± 8.688 |
| large | 7.071 ± 4.137 | 3.684 ± 3.612 | 7.049 ± 4.650 | 4.072 ± 4.422 |

数据来源：`results/task4a/position_summary.csv` 与 `results/task4b/e986df2de3cb8278/position_summary.csv`。关节误差在 `results/task4a/joint_summary.csv`，视频结果部分也应覆盖。

若要对比 FK 和 Webots 的差异，应使用相同的前 100 个样本：4.B 汇总中的 `matched_fk_mean_m`、`matched_fk_std_m` 提供该配对 FK 统计。不要把完整 500 行与前 100 行的均值差全部解释成仿真误差。

Task 1 的现有 FK 验证覆盖 Ned 3 个、SCARA 8 个测量；历史 CSV 没有逐条保存实际关节和稳定状态。不能将这部分描述成对整个工作空间的完整验证。Task 4.A 的 Ned small/medium/large 分别有 2/3/3 个非正肘角预测，已保留在统计中；它们也不等同于 Webots 移动失败。

## 4. 视频要满足哪些要求

依据 PDF 第 23–25 页：总时长不超过 10 分钟；MP4 或 MKV；所有旁白配字幕；每位组员必须参与并讲话；展示两种机械臂、多个目标和 ANN 到电机的过程；说明实现、结果及限制。整个带旁白视频不能整体加速。无旁白的机器人运动素材可加速后剪入，建议注明播放速度。

建议做成约 9 分 30 秒，留出余量。下表是拍摄清单，不是可直接朗读的讲稿：

| 时间 | 画面和操作 | 小组需要自行准备的说明 |
| --- | --- | --- |
| 0:00–0:40 | 项目名称、成员、两臂画面 | 项目目标与位置控制范围 |
| 0:40–1:40 | 流程示意、FK 命令、数据文件前几行 | FK 与 IK 的区别，数据来源和坐标单位 |
| 1:40–3:00 | 网络结构、模型 JSON、推理终端 | 标识输入、归一化、训练方式及设计理由 |
| 3:00–4:15 | Ned 连续运动到至少 3–5 个目标，保留 Console | 对应的模型、目标编号和稳定测量过程 |
| 4:15–5:30 | SCARA 连续运动到至少 3–5 个目标，拍清伸缩轴 | 三个输出如何映射到不同类型电机 |
| 5:30–7:30 | 位置、关节误差表或图，展示六组完成状态 | 样本数、单位、统计定义和有证据的结果比较 |
| 7:30–9:00 | 原始样本、日志或困难样本 | 小组自行分析误差、限制、失败或不足及原因 |
| 9:00–9:30 | 项目结构与运行入口 | 小组自行总结成果和经验 |

不必在视频中等待整个数据生成、500 轮训练或 600 次仿真，也不必逐行展示所有代码。应展示关键输入、操作、输出和可复核证据。现有批量世界没有用于交互控制的红色目标球，不要将拖动球控制或在线纠偏描述为已实现功能。

## 5. 终端里运行什么、分别展示什么

先在原项目根目录打开 PowerShell：

```powershell
Set-Location -LiteralPath 'C:\Users\77935\Desktop\智能机器人RO\cw1\nn-ik-master_now'
```

下列命令适合录屏，使用现有模型，无需重新训练：

| 命令 | 实现什么 | 建议拍摄的画面 |
| --- | --- | --- |
| `python scripts/fk_ned.py -0.54 0.4 0.9` | Ned 关节值转换成 TCP 坐标 | 命令输入与打印的 xyz |
| `python scripts/fk_scara.py 0.3 0.5 -0.1` | SCARA 关节值转换成 TCP 坐标 | rad、rad、m 的输入及 xyz |
| `python scripts/predict_ann.py ned 0.09734 0.16239 0.15728` | 大型共享模型预测 Ned 关节 | 归一化输出及电机名称/单位 |
| `python scripts/predict_ann.py scara 0.53558 0.27547 0.082` | 同一个大型模型预测 SCARA 关节 | 特别指出第三关节单位为 m |
| `python scripts/verify_task3.py` | 核对模型、选模记录、验证误差 | 三个模型的 PASS |
| `python scripts/verify_task4a.py` | 独立复算已保存的 FK 测试结果 | 六组各 500 行 PASS |
| `python scripts/verify_task4b_plan.py` | 核对前 100 行及 600 条指令 | 与 Task 4.A 预测一致的检查结果 |
| `python scripts/summarize_task4b.py` | 汇总已保存的 Webots 实测 | 六组 100/100 与 TASK4B COMPLETE |

可直接显示数据样例：

```powershell
Get-Content datasets/ned_train_1000.csv -TotalCount 6
Get-Content datasets/scara_train_1000.csv -TotalCount 6
```

这些行的坐标是输入，后三列归一化关节是监督标签。两条推理示例使用手动姿态的四舍五入 FK 坐标，并非正式评估样本，也不是宣称网络可精确逆转的证明。

下面是复现整个流水线的命令，讲解用途即可；已有结果的录屏不必执行默认训练命令：

```powershell
# 新生成演示数据到另一个目录，不覆盖正式数据
python scripts/generate_datasets.py --output demo_datasets --seed 42

# 如需演示训练，另存模型和日志，不覆盖正式选模结果
python scripts/train_ann.py --models demo_models --results results/demo_training

# 如需重新计算完整 Task 4.A，另存结果
python scripts/evaluate_ann.py --output results/task4a_demo
```

上述数据生成和训练均为程序实际计算，不应手写、修改或虚构评估数值。录完额外训练素材后，正式结果仍指向原来的 models 和 results/task3。

## 6. 重新录制机械臂运动：先建立演示副本

**重要：原项目 Task 4.B 已经完成，再运行会跳过成功样本。** 不要删除原来的 600 次记录来强制重演。复制项目后，只在副本中将既有测量目录改名保留，让副本从头执行。以下命令只准备副本，不会替你启动 Webots。

关闭可能正在写入结果的仿真。在 PowerShell 中整段运行一次：

```powershell
$sourceProject = 'C:\Users\77935\Desktop\智能机器人RO\cw1\nn-ik-master_now'
$videoProject = 'C:\Users\77935\Desktop\智能机器人RO\cw1\nn-ik-master_video'
if (Test-Path -LiteralPath $videoProject) {
    throw '演示副本已经存在。请使用该副本，或将 videoProject 改成另一个新目录名。'
}
Copy-Item -LiteralPath $sourceProject -Destination $videoProject -Recurse -ErrorAction Stop
$videoPlan = Get-Content -LiteralPath (Join-Path $videoProject 'results\task4b\plan.json') -Raw | ConvertFrom-Json
if ($videoPlan.plan_id -notmatch '^[0-9a-f]{16}$') { throw '计划 ID 格式异常，停止操作。' }
$videoResults = Join-Path $videoProject ('results\task4b\' + $videoPlan.plan_id)
$resolvedVideo = (Resolve-Path -LiteralPath $videoProject).Path
$resolvedResults = (Resolve-Path -LiteralPath $videoResults).Path
if (-not $resolvedResults.StartsWith($resolvedVideo + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw '结果目录不在演示副本中，停止操作。'
}
Rename-Item -LiteralPath $resolvedResults -NewName ($videoPlan.plan_id + '_original_evidence') -ErrorAction Stop
Set-Location -LiteralPath $videoProject
python scripts/verify_task4b_plan.py
```

这一步复制所有资源，保留了副本中的原始证据，并让默认测量路径暂时不存在。Webots 会在副本中重新创建测量目录；原项目没有变化。无需重跑 `prepare_task4b.py`，因为代码、模型和数据内容没有变化。

## 7. Webots 录屏的具体步骤

1. 启动 Webots R2023b，通过 File → Open World 打开**演示副本**中的 `worlds/task4b_ned.wbt`。核对文件路径含 `nn-ik-master_video`。
2. 等待资源加载，保持仿真暂停，调整视角，让机器人全身和底部 Console 都清楚可见。
3. 使用你已有的录屏工具录制 Webots 窗口。先录几秒检查画面、分辨率和文字是否清晰，再录正式素材。
4. 点击正常速度运行，以便看到连续运动。自动执行顺序是 small 的 100 个目标、medium 的 100 个目标、large 的 100 个目标。
5. 至少录到 3–5 个不同目标及相应 `settled` 日志。不要把初始 home 动作当作 ANN 目标示例。若想拍摄 medium/large，先等待程序进入对应区段，再开始录制片段；长等待可使用 Fast。
6. 可以暂停并停止录屏。若要展示这次重新演示的完整完成状态，就继续到 `TASK4B DONE ned`。若只录了部分动作，明确这是演示片段，不将它作为完整复测。
7. 用同样方式打开演示副本的 `worlds/task4b_scara.wbt`。选能看到垂直伸缩轴的视角，录至少 3–5 个目标。
8. 对两臂，至少各录一个能看清动作的完整片段；展示目标与实际值时，可搭配对应 JSON 中的 `target_xyz_m`、`predicted_joints`、`actual_joints`、`measured_xyz_m`。
9. 正式统计仍从**原项目**读取。若演示副本只运行了一部分，它显示 INCOMPLETE 是正常的，不要拿它替换原项目结果。
10. 剪辑时可缩短无旁白等待片段；有旁白的讲解保持正常速度，并加字幕。小组自行分配讲解，保证所有成员都有发言。

当前世界只显示机器人运动和 Console，不显示目标坐标标记。若需要展示具体目标，就并排展示当前样本 JSON，或在后期用真实坐标做标注。不要假称画面中存在未实现的可拖动目标控制。

## 8. 结果部分和提交前检查

建议准备三类图表或表格：两臂与三模型的 Task 4.A 位置误差；18 组关节误差（分开 rad 与 m²）；同一前 100 样本的 FK/Webots 位置误差比较。误差条注明标准差，样本数与单位必须标出。没有绘图时可以展示清晰的汇总表，不要直接塞入密密麻麻的 3000 行 CSV。

小组自行回答并写成最终讲解：为什么采用这些数据规模与超参数；误差随模型/机械臂如何变化；配对 FK/Webots 差异能由哪些证据解释；非正肘角、不可达输入、验证覆盖范围有什么限制；调试过程中学到了什么。不要把猜测写成已验证结论，也不要为视频制造失败数据。

录制前检查：原模型与实测记录已保留；终端文字放大；鼠标和弹窗不遮挡数据；两臂都有多目标运动；图表显示单位、样本数和标准差；所有成员发言；所有旁白有字幕；最终时长 ≤10 分钟；导出 MP4 或 MKV。

最后注意：`.gitignore` 当前忽略 `*.npz` 和 `results/`。GitHub 默认添加文件时可能漏掉已训练模型和结果。提交 ZIP 时应检查这些文件确实包含在内；不能将“源代码上传成功”等同于“完整可复现项目已提交”。
