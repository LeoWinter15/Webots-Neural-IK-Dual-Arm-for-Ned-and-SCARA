# nn-ik - ANN-based inverse kinematics coursework

Project structure and how to run each part (Python 3, Webots R2023b;
libraries used: standard library + numpy + scikit-learn only).

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
