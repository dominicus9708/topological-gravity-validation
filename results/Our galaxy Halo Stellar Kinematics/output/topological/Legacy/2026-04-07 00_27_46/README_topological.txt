Our galaxy Halo Stellar Kinematics - topological stage v1

What this stage does
- Reads the fixed final input used by skeleton/standard.
- Computes local position-based structural dimension D_loc_pos.
- Computes local kinematic structural dimension D_loc_kin.
- Combines them into D_loc using quality-derived lambda_pos.
- Computes shell-based halo background dimension D_halo_shell.
- Computes sigma_local = D_loc - D_halo_shell.
- Computes shell-based topological_gradient_term and topological_prediction_norm.

What this stage does not do
- It does not re-filter or re-select the final input.
- It does not insert an arbitrary coupling constant.
- topological_prediction_norm is a comparison-oriented normalized form.
