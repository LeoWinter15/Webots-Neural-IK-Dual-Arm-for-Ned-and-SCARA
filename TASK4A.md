# Task 4.A - held-out ANN evaluation using FK

See [the Chinese guide](TASK4A_zh.md) for detailed usage. This implements
Coursework_1_2627.pdf page 12 with Python 3, NumPy and the standard library.
No Webots session is needed.

From the project root:

```console
python scripts/evaluate_ann.py
python scripts/verify_task4a.py
python -m unittest discover -s tests -v
```

Evaluation has already been run for the current models. It loads the frozen
Task 3 small/medium/large checkpoints, predicts both arms' complete 500-row
test sets, denormalises through `robot_config.py`, and applies Task 1 FK.
There are 3000 predictions in total. No fitting, tuning, clipping, analytical
IK fallback, sample filtering or resampling occurs.

The exact PDF metrics are:

- Position: Euclidean distance between predicted and target TCP, in metres.
- Revolute joints: **signed** `atan2(sin(true-pred), cos(true-pred))`, in radians.
- Prismatic joint: **squared** `(true-pred)**2`, in **m^2** (SCARA output 3 only).
- Each error has its own mean and population standard deviation (`ddof=0`).
  Signed angular errors are not made absolute; prismatic statistics are not RMSE.

`results/task4a/` contains:

- `position_summary.csv`: six model/arm rows with mean/std in metres and mm.
- `joint_summary.csv`: eighteen model/arm/joint rows with named joints and units.
- `small_ned_samples.csv`, etc.: six 500-row files containing reference/predicted
  positions and joints, all errors and nonpositive-elbow/physical-limit flags.
- `summary.json`: all metrics, sample counts, source/model/data hashes and policy.
- `verification.json`: independent CSV/FK/error/statistics recomputation checks.

Files preserve test order with 1-based sample indices. Nonpositive elbow
predictions remain in the statistics. Test hashes, counts, uniqueness, reference
FK consistency, model/source identity and train/validation separation are checked.
Training/validation data are read solely for integrity/overlap checks.

Default reruns overwrite same-named outputs. Use `--output results/task4a_run2`
for another destination, and `--results results/task4a_run2` for verification.
The first 100 original test rows remain available for the later Task 4.B Webots
evaluation, which is not performed here. These are computed technical records,
not final poster/video interpretation or the group's reflective discussion.
