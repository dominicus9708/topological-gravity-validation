Our galaxy Halo Stellar Kinematics - topological stage 012

Changes in this revision
- Robust structural aggregation is retained.
- Standard comparison is read from standard shell summary model files.
- Standard shell stream-overlap diagnostics are ingested from standard stage 004.
- sigma_local_mean_shell is retained for reference but no longer used as the main local structural shell term.
- A shell-local structural spread is defined as sigma_local_spread_shell = weighted MAD of (D_loc - D_halo_shell).
- The effective shell contrast is Sigma_shell_effective = Sigma_bg_shell + lambda * sigma_local_spread_shell.
- The main topological gradient diagnostic is now computed from Sigma_shell_effective.
- c_info is not re-applied inside the structural spread term; it remains reserved for later physical response translation.
- shell_caution_label is retained to distinguish sparse / stream-rich / nominal shells.
- Coupling remains diagnostic only and is not applied to prediction.
