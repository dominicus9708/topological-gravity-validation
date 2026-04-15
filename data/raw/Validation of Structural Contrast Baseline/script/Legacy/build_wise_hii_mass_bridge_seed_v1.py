#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

def main() -> int:
    out_dir = (
        DEFAULT_PROJECT_ROOT
        / "data"
        / "raw"
        / "Validation of Structural Contrast Baseline"
        / "wise_hii_catalog"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_template = pd.DataFrame([
        {"wise_name":"G028.983-00.604","object_type":"Galactic H II region","mass_bridge_status":"needs_followup","preferred_mass_type":"equivalent_ionizing_star_mass","mass_value_msun":"","mass_value_min_msun":"","mass_value_max_msun":"","mass_source_short":"","mass_source_url":"","mass_source_note":"Need dedicated ionizing-source or radio-continuum based bridge source.","recommended_next_step":"Search radio continuum / Lyman-continuum or identified ionizing source paper."},
        {"wise_name":"G029.956-00.020","object_type":"Galactic H II region","mass_bridge_status":"needs_followup","preferred_mass_type":"equivalent_ionizing_star_mass","mass_value_msun":"","mass_value_min_msun":"","mass_value_max_msun":"","mass_source_short":"","mass_source_url":"","mass_source_note":"Need dedicated ionizing-source or radio-continuum based bridge source.","recommended_next_step":"Search radio continuum / Lyman-continuum or identified ionizing source paper."},
        {"wise_name":"G034.256+00.136","object_type":"Galactic H II region","mass_bridge_status":"radio_proxy_available","preferred_mass_type":"equivalent_ionizing_star_mass","mass_value_msun":"","mass_value_min_msun":"","mass_value_max_msun":"","mass_source_short":"Campbell-White et al. 2018 Table A1","mass_source_url":"https://arxiv.org/pdf/1804.04937","mass_source_note":"Has 1.4 GHz flux and log N_Ly in Table A1; suitable for equivalent ionizing-star mass bridge.","recommended_next_step":"Convert log N_Ly to equivalent O/B-star mass or cluster-ionizing proxy."},
        {"wise_name":"G045.475+00.130","object_type":"Galactic H II region","mass_bridge_status":"radio_proxy_available","preferred_mass_type":"equivalent_ionizing_star_mass","mass_value_msun":"","mass_value_min_msun":"","mass_value_max_msun":"","mass_source_short":"Campbell-White et al. 2018 Table A1","mass_source_url":"https://arxiv.org/pdf/1804.04937","mass_source_note":"Has 1.4 GHz flux and log N_Ly in Table A1; suitable for equivalent ionizing-star mass bridge.","recommended_next_step":"Convert log N_Ly to equivalent O/B-star mass or cluster-ionizing proxy."},
        {"wise_name":"G060.881-00.135","object_type":"Galactic H II region","mass_bridge_status":"needs_followup","preferred_mass_type":"equivalent_ionizing_star_mass","mass_value_msun":"","mass_value_min_msun":"","mass_value_max_msun":"","mass_source_short":"","mass_source_url":"","mass_source_note":"Need dedicated ionizing-source or radio-continuum based bridge source.","recommended_next_step":"Search radio continuum / Lyman-continuum or identified ionizing source paper."},
    ])

    sources = pd.DataFrame([
        {"wise_name":"G028.983-00.604","source_role":"primary_catalog","source_short":"WISE Catalog of Galactic H II Regions (Table2 / V1.5)","source_url":"https://astro.phys.wvu.edu/wise/Table2.txt","supported_fields":"WISE identifier, class K, angular radius, HII identifier, VLSR reference code","source_note":"Base catalog source only; no direct stellar mass field."},
        {"wise_name":"G029.956-00.020","source_role":"primary_catalog","source_short":"WISE Catalog of Galactic H II Regions (Table2 / V1.5)","source_url":"https://astro.phys.wvu.edu/wise/Table2.txt","supported_fields":"WISE identifier, class K, angular radius, HII identifier, VLSR reference code","source_note":"Base catalog source only; no direct stellar mass field."},
        {"wise_name":"G034.256+00.136","source_role":"primary_catalog","source_short":"WISE Catalog of Galactic H II Regions (Anderson et al. 2014)","source_url":"https://astro.phys.wvu.edu/wise/wise_catalog_hii_regions.pdf","supported_fields":"Catalog identity and H II region baseline data","source_note":"Use as catalog anchor."},
        {"wise_name":"G034.256+00.136","source_role":"radio_proxy_and_cluster_candidate","source_short":"Campbell-White et al. 2018, Table A1 / MWSC note","source_url":"https://arxiv.org/pdf/1804.04937","supported_fields":"1.4 GHz flux, log N_Ly, distance, positional coincidence with BDSB_127","source_note":"Best current bridge source among the five for mass-proxy work."},
        {"wise_name":"G045.475+00.130","source_role":"primary_catalog","source_short":"WISE Catalog of Galactic H II Regions (Anderson et al. 2014)","source_url":"https://astro.phys.wvu.edu/wise/wise_catalog_hii_regions.pdf","supported_fields":"Catalog identity and H II region baseline data","source_note":"Use as catalog anchor."},
        {"wise_name":"G045.475+00.130","source_role":"radio_proxy_and_cluster_candidate","source_short":"Campbell-White et al. 2018, Table A1 / MWSC note","source_url":"https://arxiv.org/pdf/1804.04937","supported_fields":"1.4 GHz flux, log N_Ly, distance, positional coincidence with BDSB_136","source_note":"Best current bridge source among the five for mass-proxy work."},
        {"wise_name":"G060.881-00.135","source_role":"primary_catalog","source_short":"WISE Catalog of Galactic H II Regions (Table2 / V1.3/V1.5 online table)","source_url":"https://astro.phys.wvu.edu/wise/Table2.txt","supported_fields":"WISE identifier, class K, angular radius, HII identifier, VLSR reference code","source_note":"Base catalog source only; no direct stellar mass field."},
    ])

    radio_proxy = pd.DataFrame([
        {"wise_name":"G034.256+00.136","distance_kpc":3.5,"galactocentric_distance_kpc":5.8,"effective_radius_pc":2.08,"flux_1p4GHz_Jy":10.758,"log_Nly_s_minus_1":49.04,"stromgren_radius_pc":0.7,"dynamical_age_Myr":0.2,"cluster_candidate":"BDSB_127","source_short":"Campbell-White et al. 2018 Table A1 / cluster note","source_url":"https://arxiv.org/pdf/1804.04937"},
        {"wise_name":"G045.475+00.130","distance_kpc":7.7,"galactocentric_distance_kpc":6.2,"effective_radius_pc":1.00,"flux_1p4GHz_Jy":1.747,"log_Nly_s_minus_1":48.94,"stromgren_radius_pc":0.6,"dynamical_age_Myr":0.3,"cluster_candidate":"BDSB_136","source_short":"Campbell-White et al. 2018 Table A1 / cluster note","source_url":"https://arxiv.org/pdf/1804.04937"},
    ])

    p1 = out_dir / "wise_hii_mass_bridge_template_seed.csv"
    p2 = out_dir / "wise_hii_mass_bridge_sources_seed.csv"
    p3 = out_dir / "wise_hii_mass_bridge_radio_proxy_seed.csv"

    bridge_template.to_csv(p1, index=False, encoding="utf-8-sig")
    sources.to_csv(p2, index=False, encoding="utf-8-sig")
    radio_proxy.to_csv(p3, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - mass bridge seed builder v1")
    print("=" * 72)
    print("[OK] Created:")
    print(p1)
    print(p2)
    print(p3)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())