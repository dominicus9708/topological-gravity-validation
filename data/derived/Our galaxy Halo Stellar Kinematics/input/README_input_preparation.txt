Our galaxy Halo Stellar Kinematics - final input preparation

Processed source directory
- C:\Users\mincu\Desktop\topological_gravity_project\data\derived\Our galaxy Halo Stellar Kinematics\first_processed_v1

Output directory
- C:\Users\mincu\Desktop\topological_gravity_project\data\derived\Our galaxy Halo Stellar Kinematics\input

Produced files
- gaia_rrlyrae_5d_input.csv
- gaia_rrlyrae_6d_input.csv
- input_preparation_summary.csv
- README_input_preparation.txt

This version uses actual finalization rules:
1) external_body_flag
   - conservative LMC/SMC main-body masking by sky position
2) halo_candidate_flag
   - quality_basic_pass == True
   - external_body_flag == False
   - |gal_b_deg| >= 20
   - distance_proxy_kpc >= 5
   - metallicity <= -0.5
   - and for 6D, has_full_6d == True

Operational rule
- The saved input files are treated as final fixed input for downstream stages.
- Downstream stages should not re-filter rows.
