from pathlib import Path
import pandas as pd

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sparc"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sparc_csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_file(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            r = float(parts[0])
            v_obs = float(parts[1])
            v_err = float(parts[2])
            v_gas = float(parts[3])
            v_disk = float(parts[4])
            v_bul = float(parts[5])

            rows.append([r, v_obs, v_err, v_gas, v_disk, v_bul])

    df = pd.DataFrame(
        rows,
        columns=["r", "v_obs", "v_err", "v_gas", "v_disk", "v_bul"]
    )

    return df


def main():
    for file_path in sorted(INPUT_DIR.glob("*.dat")):
        df = convert_file(file_path)

        out = OUTPUT_DIR / file_path.name.replace("_rotmod.dat", ".csv")
        df.to_csv(out, index=False)

        print("converted:", file_path.name)


if __name__ == "__main__":
    main()
