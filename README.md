# nn-ik - ANN-based inverse kinematics coursework

Project structure and how to run each part (Python 3, Webots R2023b;
libraries used: standard library + numpy + scikit-learn only).

Task 3 is implemented in this folder using **NumPy + the standard library only**.
See [TASK3.md](TASK3.md) for training, model files, prediction, and verification.

Task 4.A evaluates the saved models on both held-out test sets using Task 1 FK.
See [TASK4A.md](TASK4A.md) or [中文说明](TASK4A_zh.md). Run
`python scripts/evaluate_ann.py`, then `python scripts/verify_task4a.py`.

Task 4.B batch worlds and controllers are prepared. See
[Task 4.B Webots 操作指南](TASK4B_zh.md). Open `worlds/task4b_ned.wbt`
and `worlds/task4b_scara.wbt` in turn and run each to completion, then run
`python scripts/summarize_task4b.py`. Real Webots measurements are required;
mocked controller tests do not complete this task.

## Task 1.A - Webots controllers

Two arm controllers that accept joint values, move the arm, wait for it
to settle, then signal the supervisor to record the measured TCP:

    controllers/ned_simple/ned_simple.py    (joint_1, joint_2, joint_3 in rad)
    controllers/scara_simple/scara_simple.py(base_arm_motor, arm_motor in rad,
                                             shaft_linear_motor in metres)

Shared supervisor (already referenced by both world files):

    controllers/tcp_pose/tcp_pose.py

Run: open `worlds/articulated.wbt` (Ned) or `worlds/scara.wbt` (SCARA) in
Webots and press play. Joint values come from controllerArgs in the world
file, or from DEFAULT_JOINTS at the top of each controller. The supervisor
appends the measured TCP to `tcp_measurements.csv` in this directory.

## Task 1.B - Forward kinematics (pure Python)

    scripts/fk_ned.py      python fk_ned.py q1 q2 q3      (rad)
    scripts/fk_scara.py    python fk_scara.py q1 q2 q4    (rad, rad, m)

Both print the TCP position in the ARM frame and expose fk() for import.

## Task 1.C - FK verification against Webots

Run the Task 1.A controllers (both arms), then:

    python scripts/compare_fk_webots.py

Compares every row of tcp_measurements.csv against the analytical FK and
reports mean/std/max error in millimetres.

## Task 2 - Dataset generation

Run from the project root (Python 3, no Webots required):

    python scripts/generate_datasets.py --output datasets --seed 42

The script calls the two Task 1.B `fk()` functions and creates 14 CSV files:
`ned_train_N.csv`, `ned_validation_N.csv`, `scara_train_N.csv`, and
`scara_validation_N.csv` for N=1000, 5000, 20000, plus `ned_test.csv` and
`scara_test.csv` (500 rows each). The sizes are total samples before the
80/20 split. Change them with `--sizes 1000 5000 20000` and change the
held-out count with `--test-size 500`. The same seed reproduces the same files.

Each CSV header is `arm,x,y,z,q1_norm,q2_norm,q3_norm`. Coordinates are
metres in the arm's frame; joint targets are scaled to [-1, 1] from the
physical normalisation limits in `scripts/robot_config.py`. Ned columns q1/q2/q3 represent
joint_1/joint_2/joint_3 (radians); SCARA columns represent base_arm_motor
(radians), arm_motor (radians), shaft_linear_motor (metres). In both arms the
elbow joint is sampled strictly positive. The same limits apply to every
split. `denormalise()` reverses the scaling for later motor commands.

For each arm, every smaller training file is a prefix of its larger training
file; the same applies to validation. Training, validation and test samples
are disjoint, with the test file reserved for Task 4. Run the command yourself
to produce the final coursework data; review the resulting values and sizes.

## Task 1/2 finalisation

### Valid measurements and failures

The arm controllers now reject settle timeouts and simulation termination before
settling. They append target joints, actual joint readings, elapsed simulation
time and status to `motion_failures.jsonl`, exit with failure, and do not request
a TCP measurement. Successful requests use an atomic flag-file write. The
supervisor rejects malformed requests instead of substituting zero joints.
The existing `tcp_measurements.csv` is preserved; its historical rows do not
contain actual joint readings or a recorded settled status.

Run `python scripts/compare_fk_webots.py` after measuring. It saves a copy of
the measurements and `summary.json` under `results/task1/<measurement-hash>/`.
The report includes per-arm count/mean/population-standard-deviation/max in
metres, tested joint values, source hashes and measurement provenance.
This is a measurement summary, not a substitute for Webots verification.

### One source for normalisation

` scripts/robot_config.py ` separates `sampling_limits` (positive elbow branch)
from `normalisation_limits` (PDF physical motor limits). Tasks 3 and 4 must
import `normalise_joints(arm, joints)` / `denormalise_joints(arm, outputs)` from
this module. Do not independently redefine these bounds. SCARA output 3 is
shaft displacement in metres; the other active joints are in radians.

Ned joint_3 is sampled in (0, 1.3] but scaled with [-1.5, 1.3]. SCARA arm_motor
is sampled in (0, 0.83] but scaled with [-0.83, 0.83]. Labels therefore need not
occupy the entire [-1, 1] interval. Do not clip inverse-normalised predictions
silently; record any output-limit policy consistently in training/evaluation.

Datasets were regenerated with the original seeds (Ned 42, SCARA 43), retaining
all original xyz coordinates and row order. Only the elbow normalised labels
changed. Any model trained on the old labels must be retrained, or explicitly
retain its old scaling; never decode it with the new configuration.

`datasets/metadata.json` records configuration version, per-arm joint names,
units, sampling/scaling bounds, seeds, sample counts, nesting/test policies,
CSV hashes and generation/FK source hashes. Copy this configuration alongside
future trained models. `datasets.zip` contains the updated CSVs and metadata.
The prior code and data are retained in `backups/before_task12_cleanup.zip`.

### Current validation coverage

The current CSV contains three Ned and eight SCARA measurements. SCARA shaft
values recorded are 0, -0.05, -0.07, -0.10 and -0.15 m. The latest snapshot
is `results/task1/ca8942e9abd2/`. Mean/max position errors are approximately
0.125/0.250 mm for Ned and 0.120/0.120 mm for SCARA.

A real SCARA settle timeout was recorded separately in `motion_failures.jsonl`
(target shaft -0.18 m; actual approximately -0.172917 m). This failed attempt is
absent from the valid measurement CSV. Keep clearance below the shaft before
Task 4.B; the full dataset includes displacements down to -0.2 m.
Negative second-joint measurements are general FK checks only; Task 2 sampling
continues to require the positive elbow branch.

The revised controller branches were checked with mocked Webots devices for
success, settle timeout and simulation termination for both arms. The updated measurements and failure log additionally provide real Webots
evidence for SCARA success and timeout handling. The Ned failure branch has
only been checked with mocked devices.
