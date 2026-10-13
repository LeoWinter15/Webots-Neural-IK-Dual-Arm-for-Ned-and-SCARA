# Task 4.B：Webots 实测操作指南

## 当前状态

已完成控制器、测试世界、指令准备和汇总程序，并生成两种机械臂各自的前 100 行测试集及 600 组模型预测电机指令。**尚需实际运行 Webots，获得 600 次 TCP 测量，Task 4.B 才算完成。** 单元测试中的模拟设备数据仅用于检查程序，没有写入真实评估结果。

本机安装的是 Webots R2023b。本步骤不需要训练模型，不需要在 Webots 的 Python 中安装 NumPy；ANN 预测已由准备脚本完成，Webots 控制器只使用 Python 标准库和 Webots API。

## 第一步：打开 Ned 测试世界

1. 启动 Webots R2023b。
2. 暂停当前仿真，选择菜单 **File → Open World…**（打开世界）。
3. 在本项目的 `worlds` 文件夹中选择 **`task4b_ned.wbt`**。
4. 若旧世界询问是否保存临时仿真状态，本次测试不需要保存旧世界的姿态修改。
5. 等待机器人模型和远程 PROTO 资源加载完毕。首次加载可能需要联网。
6. 查看底部 Console（控制台），确认没有红色资源加载错误。新世界已经指定控制器 `task4b_runner` 和参数 `ned`，不用修改控制器或关节值。

本机项目路径：`C:\Users\77935\Desktop\智能机器人RO\cw1\nn-ik-master_now`。该路径仅用于此操作说明，代码通过自身位置定位项目，未硬编码本机路径。

## 第二步：运行 Ned 的 300 次测量

1. 点击工具栏的运行按钮。可以使用 **Fast** 模式加快仿真；不确定图标时，将鼠标悬停查看提示。普通运行模式也可以。
2. 控制器先移动到固定起始姿态，然后依次执行 small、medium、large 三个模型，各 100 个目标。
3. Console 会出现类似：

```text
TASK4B START ned: plan=..., 3 models x 100 targets
TASK4B ned small 001/100 settled
TASK4B ned small 002/100 settled
...
TASK4B ned large 100/100 settled
TASK4B DONE ned: new failures=0. Results: ...
```

4. 运行时不要拖动机器人、手动修改关节、修改世界文件或更换控制器。仅调整视角即可。
5. 等待出现 **`TASK4B DONE ned`**，世界会自动暂停。运行时间取决于机器速度和仿真模式，不是每次固定几分钟。每个目标最多等待 60 秒仿真时间。
6. 此时汇总中 SCARA 显示 `missing=100` 是正常的，因为尚未运行另一种机械臂。

## 第三步：运行 SCARA 的 300 次测量

1. 选择 **File → Open World…**。
2. 打开同一个 `worlds` 文件夹中的 **`task4b_scara.wbt`**。
3. 不需要保存刚才 Ned 世界的最终关节姿态。测量结果已经单独保存为 JSON 文件。
4. 同样点击运行或 Fast。
5. 等待依次完成 small、medium、large 三组各 100 次测量。
6. 出现 **`TASK4B DONE scara`** 后会自动暂停。

两个测试世界的地面均下移到 z=-0.4 m，机械臂根节点位置没有改变，基座固定。这样可为 SCARA 向下伸出的轴提供空间，避免之前地面接触导致的超时。实测使用相对于 ARM 根节点的坐标，不把地面偏移加到 TCP 结果中。原始 `articulated.wbt`、`scara.wbt` 未修改。

## 第四步：检查全部完成

两种机械臂每次跑完都会自动更新汇总。也可以在 PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'C:\Users\77935\Desktop\智能机器人RO\cw1\nn-ik-master_now'
python scripts/summarize_task4b.py
```

如果系统中的 `python` 不能运行，可以使用当前机器上已验证的解释器：

```powershell
& 'D:\Anaconda3\python.exe' scripts/summarize_task4b.py
```

完整成功应显示六行 `settled=100/100 failed=0 missing=0`，最后显示：

```text
TASK4B COMPLETE
```

只有以上条件全部满足，六组完整的实测位置误差均值和标准差才会填写。若出现 `TASK4B INCOMPLETE`，请查看哪些机械臂或模型还缺少测量，或存在超时。

完成后可以直接告诉我“两个世界都运行完了”，我会读取本地结果进一步检查，不需要手工复制所有数据。

## 中断、超时和其他报错

- **中途关闭 Webots或停止控制器**：已经完成的样本逐条保存。重新加载同一个测试世界，再点击运行，即可跳过成功样本，继续缺失样本并重试失败样本。单纯暂停后继续运行也可以，不必重启。
- **已自动暂停，想重试失败样本**：重新打开同一个 `.wbt` 文件，确认重新加载世界，然后运行。仅按播放不会重新启动已经结束的控制器。
- **`settle_timeout`**：该次移动未在 60 秒仿真时间内达到稳定条件，记录为失败，不混入成功位置误差统计。控制器继续处理后续样本。等本轮结束后，可重新加载世界重试。若持续失败，将 Console 中模型、样本编号和错误信息发给我。
- **`Home pose failed`**：起始姿态无法稳定，控制器暂停。请保留错误信息，先排查碰撞或电机问题。
- **`Changed input/source`**：代码、模型、数据或测试世界相对已准备的计划发生变化。不要通过跳过校验继续运行。在项目根目录执行 `python scripts/prepare_task4b.py`，再重新加载测试世界。新的计划会使用新的结果目录，旧记录保留；不要把不同计划的测量混在一起。
- **`plan.json` 找不到**：执行 `python scripts/prepare_task4b.py`，然后重新加载测试世界。
- **Python 无法启动**：在 Webots 的 **Tools → Preferences → General → Python command** 中检查 Python 命令；本机可填 `D:\Anaconda3\python.exe`，应用后重新加载世界。控制器本身不需要 NumPy。
- **PROTO / 网络资源加载错误**：必须等模型完整加载后再运行。将实际错误信息发给我，不要将缺失机器人资源的场景当作有效测试。

不要同时开两个相同机械臂的测试世界写入同一计划。按上述步骤先 Ned 后 SCARA 即可。

## 程序做了什么

`scripts/prepare_task4b.py` 已执行。它从原始 `ned_test.csv`、`scara_test.csv` 精确取前 100 行，生成 `datasets/ned_test_100_task4b.csv` 和 `datasets/scara_test_100_task4b.csv`；加载三个冻结的 Task 3 模型，将预测物理关节值写入 `results/task4b/plan.json`。不重新抽样、训练或选模，也不裁剪预测。

`task4b_runner` 既是机械臂控制器，又具有 Supervisor 权限。它使用电机命令驱动机器人，然后检查三个活动关节及固定关节的位置传感器。各关节必须连续 20 个仿真步处于容差内：旋转关节 0.002 rad，SCARA 直线关节 0.0005 m。固定关节目标为零。世界基本时间步为 16 ms。

稳定后，通过 `TCP.getPose(ARM)` 读取仿真中的 TCP 相对坐标；不会用 FK 结果冒充实测。这里采用同一个具有 Supervisor 权限的控制器完成驱动与读取，避免跨控制器文件握手造成的样本对应问题。API 依据 [Webots Supervisor 官方文档](https://github.com/cyberbotics/webots/blob/R2023b/docs/reference/supervisor.md)。

每组误差按 `sqrt((x_measured-x_target)^2 + (y_measured-y_target)^2 + (z_measured-z_target)^2)` 计算，以米为单位。均值和总体标准差（`ddof=0`）分别统计，共 3 模型 × 2 机械臂 = 6 组。Task 4.B 不要求关节误差统计，但保留实际传感器值作为稳定性证据。

## 结果文件位置和含义

`results/task4b/plan.json` 中的 `plan_id` 对应结果子目录 `results/task4b/<plan_id>/`。每次准备相同输入得到相同 ID；改变模型、代码或世界会使用新 ID。

- `ned/`、`scara/`：每个样本的当前结果，包括预测关节、实际传感器读数、目标坐标、实测坐标、耗时与状态。
- `ned/attempts/`、`scara/attempts/`：保留每次尝试，包括被后续重试替代的失败记录。
- `ned/sessions/`、`scara/sessions/`：运行会话和仿真时间步等记录。
- `position_summary.csv`：六组位置误差统计及完成数量。`mean_m`、`std_m` 只有在该组 100 个样本全部稳定测量后才填写。
- `summary.json`：机器可读汇总；`all_complete=true` 表示六组全部完成。
- `samples.csv`：600 个计划样本的当前状态及实测结果；未测量项明确标为 `missing`。

未完成时，`settled_subset_mean_m` 和 `settled_subset_std_m` 仅表示当前成功子集，不是完整 100 样本的正式结果。失败样本不会静默丢弃。`matched_fk_mean_m/std_m` 使用相同样本的 FK 预测位置作为配对比较；`webots_vs_fk_mean_m` 是实测 TCP 与该 FK 位置的平均距离。它们与完整 500 样本的 Task 4.A 汇总有不同样本范围。

## 自动化代码检查

```console
python -m unittest discover -s tests -v
```

新增检查覆盖连续稳定判定、超时、仿真提前结束、Supervisor 坐标提取、计划标识、缺失与失败统计、完整 100 样本要求，以及传感器容差检查。这些是程序验证，不是 Webots 实测证据。
