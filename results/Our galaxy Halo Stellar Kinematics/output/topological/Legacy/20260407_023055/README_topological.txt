Our galaxy Halo Stellar Kinematics - topological stage 003

Changes in this revision
- Output is written into a timestamped subfolder.
- Background reference is written as D_bg = 3.0.
- Main topological driver uses shell-averaged background contrast:
    Sigma_bg_shell = D_halo_shell - D_bg
- Main gradient term uses d(Sigma_bg_shell)/dr only.
- pointwise sigma_local = D_loc - D_halo_shell is retained only as a diagnostic.
- shell-by-shell xi*c_info^2 inverse estimates are retained only as diagnostics.
- the main prediction adopts a single global A = xi*c_info^2 estimated from stable 6D shells,
  then projects that A into both 6D and 5D shell summaries.
- Input rows are not re-filtered or re-selected.

Structural reference
- D_bg = 3.0

Stabilization
- MIN_PARALLAX_MAS = 0.02
- MAX_DISTANCE_PROXY_KPC = 50.0
- MAX_VT_PROXY_KMS = 1000.0
- MIN_SHELL_N_FOR_GRADIENT = 20
- MIN_SHELL_N_FOR_GLOBAL_A = 20
