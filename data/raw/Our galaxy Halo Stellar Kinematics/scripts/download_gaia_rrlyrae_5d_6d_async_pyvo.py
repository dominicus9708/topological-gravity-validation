from __future__ import annotations
import io
import time
import pathlib
import requests
import pyvo

# 이 파일 위치:
# <project_root>\data\raw\Our Galaxy Halo Stellar Kinematics\download_gaia_rrlyrae_5d_6d_async_pyvo.py

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]

RAW_BASE = PROJECT_ROOT / "data" / "raw" / "Our Galaxy Halo Stellar Kinematics"
FOLDER_5D = RAW_BASE / "gaia_rrlyrae_5d"
FOLDER_6D = RAW_BASE / "gaia_rrlyrae_6d"

SERVICE_URL = "https://gaia.aip.de/tap"

QUERY_5D = """
SELECT
    v.source_id,
    s.ra,
    s.dec,
    s.parallax,
    s.parallax_error,
    s.pmra,
    s.pmra_error,
    s.pmdec,
    s.pmdec_error,
    s.phot_g_mean_mag,
    s.bp_rp,
    v.int_average_g,
    v.metallicity
FROM gaiadr3.vari_rrlyrae v
JOIN gaiadr3.gaia_source s
  ON v.source_id = s.source_id
"""

QUERY_6D = """
SELECT
    v.source_id,
    s.ra,
    s.dec,
    s.parallax,
    s.parallax_error,
    s.pmra,
    s.pmra_error,
    s.pmdec,
    s.pmdec_error,
    s.phot_g_mean_mag,
    s.bp_rp,
    v.int_average_g,
    v.metallicity,
    s.radial_velocity,
    s.radial_velocity_error
FROM gaiadr3.vari_rrlyrae v
JOIN gaiadr3.gaia_source s
  ON v.source_id = s.source_id
WHERE s.radial_velocity IS NOT NULL
"""

README_5D = """Our Galaxy Halo Stellar Kinematics - 5D raw

Purpose
- Store the practically downloaded Gaia DR3 RR Lyrae 5D raw CSV.
"""

README_6D = """Our Galaxy Halo Stellar Kinematics - 6D raw

Purpose
- Store the practically downloaded Gaia DR3 RR Lyrae 6D raw CSV.
"""

def save_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def fetch_csv_async(query: str, runid: str) -> str:
    session = requests.Session()
    tap_service = pyvo.dal.TAPService(SERVICE_URL, session=session)

    job = tap_service.submit_job(query, language="ADQL", runid=runid, queue="5m")
    job.run()
    print(f"[INFO] Submitted job {runid}: {job.url}")

    while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
        print(f"[INFO] {runid} phase: {job.phase} ... waiting")
        time.sleep(30.0)
        job = pyvo.dal.AsyncTAPJob(job.url, session=session)

    print(f"[INFO] {runid} final phase: {job.phase}")
    job.raise_if_error()

    result = job.fetch_result()
    table = result.to_table()

    buf = io.StringIO()
    table.write(buf, format="ascii.csv", overwrite=True)
    return buf.getvalue()

def main() -> None:
    FOLDER_5D.mkdir(parents=True, exist_ok=True)
    FOLDER_6D.mkdir(parents=True, exist_ok=True)

    save_text(FOLDER_5D / "README_raw.txt", README_5D)
    save_text(FOLDER_6D / "README_raw.txt", README_6D)
    save_text(FOLDER_5D / "adql_query.txt", QUERY_5D.strip() + "\n")
    save_text(FOLDER_6D / "adql_query.txt", QUERY_6D.strip() + "\n")

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] 5D output folder: {FOLDER_5D}")
    print(f"[INFO] 6D output folder: {FOLDER_6D}")

    csv_5d = fetch_csv_async(QUERY_5D, "our_galaxy_rrlyrae_5d")
    save_text(FOLDER_5D / "gaia_rrlyrae_5d_raw.csv", csv_5d)

    csv_6d = fetch_csv_async(QUERY_6D, "our_galaxy_rrlyrae_6d")
    save_text(FOLDER_6D / "gaia_rrlyrae_6d_raw.csv", csv_6d)

    print("[DONE] Saved:")
    print(FOLDER_5D / "gaia_rrlyrae_5d_raw.csv")
    print(FOLDER_6D / "gaia_rrlyrae_6d_raw.csv")

if __name__ == "__main__":
    main()