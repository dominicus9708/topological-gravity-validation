import os
import pandas as pd

INPUT_DIR = "data/raw/sparc"
OUTPUT_DIR = "data/raw/sparc_csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_file(path):

    rows = []

    with open(path, "r") as f:
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

    for fname in os.listdir(INPUT_DIR):

        if not fname.endswith(".dat"):
            continue

        path = os.path.join(INPUT_DIR, fname)

        df = convert_file(path)

        out = os.path.join(
            OUTPUT_DIR,
            fname.replace("_rotmod.dat", ".csv")
        )

        df.to_csv(out, index=False)

        print("converted:", fname)


if __name__ == "__main__":
    main()