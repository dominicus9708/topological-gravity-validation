Our galaxy Halo Stellar Kinematics - topological stage 007

Changes in this revision
- Output is written into a timestamped subfolder.
- Final input remains fixed and is not re-filtered.
- Background reference is written as D_bg = 3.0.
- Hard clipping of proxy values is reduced.
- Stabilization is moved toward quality-weighted robust aggregation.
- Local pointwise sigma is retained for diagnostics only.
- Coupling-related quantities are retained only as diagnostics.

Stabilization logic in 005
- Positive-parallax proxy use is retained, but hard max clipping is reduced.
- Position and velocity features use soft asinh compression before robust scaling.
- Shell structural estimates use weighted winsorized aggregation.
- Coupling diagnostics are reported only when shell support is sufficient.

Parameters
- MIN_SHELL_N_FOR_GRADIENT = 20
- MIN_SHELL_N_FOR_COUPLING_DIAG = 20
- WINSOR_Q_LOW = 0.1
- WINSOR_Q_HIGH = 0.9
- SOFT_WEIGHT_POWER = 2.0
