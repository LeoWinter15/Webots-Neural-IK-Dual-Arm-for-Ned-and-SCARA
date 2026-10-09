# Task 3 - shared ANN inverse kinematics

Implementation follows Coursework_1_2627.pdf, pages 7 and 10-11.
Run commands below from this project directory. Python 3 and NumPy are the
only dependencies; Webots is not needed for training. Existing Task 1/2
controllers, FK implementations, datasets and joint scaling are reused.

## Train all three models

```console
python scripts/train_ann.py
```

| Model | Task 2 size per arm before split | Combined training rows | Combined validation rows | Output |
| --- | ---: | ---: | ---: | --- |
| small | 1000 | 1600 | 400 | models/ik_shared_1000.npz |
| medium | 5000 | 8000 | 2000 | models/ik_shared_5000.npz |
| large | 20000 | 32000 | 8000 | models/ik_shared_20000.npz |

Each model is one shared network fitted to BOTH arms. Small/medium/large
denote training dataset sizes, not different network widths. Training reads
the 12 named training/validation CSVs and metadata.json. Test CSVs are never
opened, used to fit preprocessing, used to choose parameters, or evaluated.

Before training, the script checks file hashes, row counts, finite values,
normalisation configuration, positive elbow labels, nested splits, and
train/validation position separation. Test-set disjointness is inherited from
Task 2's generation/provenance; Task 3 does not reopen the test sets to check it.

## Network and training settings

- Inputs: `[is_scara, is_ned, x, y, z]`. Ned uses `[0, 1]`, SCARA `[1, 0]`.
  Coordinates are in metres in each robot's ARM frame, not Webots world coordinates.
- Combined training xyz alone determine coordinate means and population standard
  deviations. The saved scaler is reused for validation and prediction. Flags
  remain exactly 0 or 1.
- Architecture: `5 -> 64 -> 64 -> 3`, with tanh at every layer, 4739 parameters.
  Xavier uniform weights and zero biases; float64 calculations.
- Outputs: three joint values normalised with the physical limits in
  `scripts/robot_config.py`. The output-layer tanh bounds them to `[-1, 1]`.
- Loss: mean squared error over all rows and all three normalised outputs.
  Both arms have equal row counts and equal weight in the selection metric.
- Optimiser: Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, minibatch size 256.
- Candidate learning rates: 0.001 and 0.003; each starts from the same seed 42
  and receives the same deterministic minibatch permutations (shuffle seed 43).
- At most 500 epochs per candidate. Early stopping uses the explicit combined
  validation set, patience 60 epochs, absolute improvement threshold 1e-7.
  The absolute lowest validation-MSE checkpoint is retained, including
  improvements smaller than the patience threshold.
- For each dataset size, save the candidate with the lowest validation MSE.
  Validation is used for selection, so these numbers are not unbiased final
  performance estimates. Final held-out evaluation belongs to Task 4.

The settings are configurable, for example:

```console
python scripts/train_ann.py --hidden 64 64 --learning-rates 0.001 0.003 --epochs 500 --batch-size 256 --patience 60 --seed 42 --models models_run2 --results results/task3_run2
```

Default reruns replace the three default model files and current history files.
Use separate `--models` and `--results` directories to preserve another experiment.
The seed reproduces runs on the same numerical environment; other NumPy/BLAS
versions can introduce floating-point differences. Python/NumPy versions and
source/data SHA-256 hashes are recorded with the models.

## Saved evidence and model format

- `models/ik_shared_N.npz`: numeric weights, biases, scaler and embedded JSON
  metadata. Uses `allow_pickle=False` when loading; no sklearn/pickle dependency.
- `models/ik_shared_N.json`: human-readable copy of the metadata: full robot
  configuration, hyperparameters, selected learning rate/epoch, both candidates,
  validation metrics and source/data hashes.
- `results/task3/*_candidate_*_history.csv`: epoch-by-epoch training MSE,
  combined validation MSE, and separate Ned/SCARA validation MSE.
- `results/task3/summary.json`: selected models, hashes, timing and all trials.

Per-arm validation metrics include normalised MSE, mean absolute error for each
output, and the number of predictions with a nonpositive physical elbow.
Loading is checked against the current robot configuration, input/output order,
layer shapes and finite parameters. Training reloads each selected model and
checks exact prediction equality on both arms' validation inputs.

## Predict motor targets

```console
python scripts/predict_ann.py ned 0.09734 0.16239 0.15728
python scripts/predict_ann.py scara 0.53558 0.27547 0.082 --model models/ik_shared_5000.npz
```

These are rounded FK positions for manual poses (Ned: -0.54, 0.4, 0.9 rad;
SCARA: 0.3 rad, 0.5 rad, -0.1 m), not held-out evaluation samples. The default
model is the large model. Outputs show normalised values and actual motor names
with their units. The API for a later Task 4 script in `scripts/` is:

```python
from pathlib import Path
from ann_model import IKModel

root = Path(__file__).resolve().parent.parent
model = IKModel(root / "models" / "ik_shared_20000.npz")
xyz = [[0.53558, 0.27547, 0.082]]  # N x 3 array, ARM frame, metres
q_norm = model.predict_normalised("scara", xyz)
q_motor = model.predict_joints("scara", xyz)
```

| Output | Ned motor / units | SCARA motor / units |
| --- | --- | --- |
| 1 | joint_1 / rad | base_arm_motor / rad |
| 2 | joint_2 / rad | arm_motor / rad |
| 3 | joint_3 / rad | shaft_linear_motor / m |

Physical values are obtained through `denormalise_joints` from the existing
shared configuration. No post-prediction clipping, elbow projection, analytical
IK fallback or workspace projection is applied. Physical joint bounds follow
from tanh, but a positive elbow is not mathematically guaranteed: the labels
only cover the positive branch of wider physical limits. The CLI reports a
nonpositive elbow; validation records its frequency. Predictions for unreachable
or out-of-distribution points are not guaranteed to be meaningful. Task 4 must
use this same output policy consistently or explicitly record any change.

## Verification

```console
python -m unittest discover -s tests -v
python scripts/verify_task3.py
```

Checks include central finite-difference verification of every parameter's
gradient in a tiny network, Adam's first update and loss reduction, model
round-trip predictions, physical scaling for both arms, configuration rejection,
flags/bounds/input validation, train/validation provenance and test-split rejection.
These tests use synthetic arrays and existing train/validation files only.

After training, `verify_task3.py` additionally checks all three model hashes,
source hashes, metadata copies, selected candidates/epochs against the recorded
loss histories, and recomputes validation MSE from the saved models. It writes
`results/task3/verification.json`. It does not evaluate the Task 4 test sets.

The current machine's scikit-learn import fails due to a NumPy compatibility
issue. Task 3 does not import it and does not require changing the environment.
Execution on GRID / another computer and Webots-based validation remain separate
project-level checks required before final submission.

## Coursework scope

This adds training and prediction for Task 3. Task 4 test-set/FK/Webots evaluation
and Task 5 deliverables are not performed here. Training histories are technical
records, not poster/video conclusions or the group's AI reflective statement.
Retain this chat as a record of AI-assisted code generation and automated checks;
the group still needs to review and understand the submitted implementation.
