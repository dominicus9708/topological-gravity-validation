Our galaxy Halo Stellar Kinematics - topological stage 004

Changes in this revision
- Output is written into a timestamped subfolder.
- Background reference is written as D_bg = 3.0.
- Main topological driver uses shell-averaged background contrast:
    Sigma_bg_shell = D_halo_shell - D_bg
- Main gradient term uses d(Sigma_bg_shell)/dr only.
- pointwise sigma_local = D_loc - D_halo_shell is retained only as a diagnostic.
- stabilization logic is retained for proxy control and shell-gradient reliability.
- the global A = xi*c_info^2 application layer is removed.
- coupling quantities are exported only as diagnostics, not as forced predictions.

Stabilization
- MIN_PARALLAX_MAS = 0.02
- MAX_DISTANCE_PROXY_KPC = 50.0
- MAX_VT_PROXY_KMS = 1000.0
- MIN_SHELL_N_FOR_GRADIENT = 20
- MIN_SHELL_N_FOR_COUPLING_DIAG = 20
