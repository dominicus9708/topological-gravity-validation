from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TABLE_DIR = BASE_DIR / "postprocess_outputs" / "tables"
FIG_DIR = BASE_DIR / "postprocess_outputs" / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV = TABLE_DIR / "merged_rotation_summary.csv"
CLASS_CSV = TABLE_DIR / "classification_counts.csv"
REP_CSV = TABLE_DIR / "representative_6_galaxies.csv"


# =========================================================
# 데이터 로드
# =========================================================
def load_data():
    if not MERGED_CSV.exists():
        raise FileNotFoundError(f"파일 없음: {MERGED_CSV}")
    if not CLASS_CSV.exists():
        raise FileNotFoundError(f"파일 없음: {CLASS_CSV}")

    merged = pd.read_csv(MERGED_CSV)
    class_df = pd.read_csv(CLASS_CSV)

    rep_df = None
    if REP_CSV.exists():
        rep_df = pd.read_csv(REP_CSV)

    return merged, class_df, rep_df


# =========================================================
# Figure 1: chi2 histogram
# =========================================================
def plot_chi2_distribution(merged: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(merged["chi2_abar"], bins=30, alpha=0.6, label=r"$\beta_{\mathrm{abar}}$")
    plt.hist(merged["chi2_formula"], bins=30, alpha=0.6, label=r"$\beta_{\mathrm{formula}}$")
    plt.xlabel(r"$\chi^2$")
    plt.ylabel("Number of galaxies")
    plt.title(r"Distribution of $\chi^2$ across the SPARC sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_chi2_distribution.png", dpi=300)
    plt.close()


# =========================================================
# Figure 2: classification bar chart
# =========================================================
def plot_classification_counts(class_df: pd.DataFrame):
    order = ["improved", "worsened", "similar"]

    class_df = class_df.copy()
    class_df["classification"] = pd.Categorical(
        class_df["classification"], categories=order, ordered=True
    )
    class_df = class_df.sort_values("classification")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(class_df["classification"], class_df["count"])
    plt.xlabel("Classification")
    plt.ylabel("Number of galaxies")
    plt.title("Galaxy-by-galaxy fit classification")

    for bar, count in zip(bars, class_df["count"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_classification_counts.png", dpi=300)
    plt.close()


# =========================================================
# Figure 3 helper: 대표 은하 추천표 저장
# =========================================================
def save_representative_summary(merged: pd.DataFrame):
    selected = merged[
        merged["galaxy"].isin(["UGC00128", "DDO154", "NGC6674", "UGC02953", "F574-2", "NGC3953"])
    ].copy()

    cols = [
        "galaxy",
        "chi2_abar",
        "chi2_formula",
        "delta_chi2",
        "improvement_ratio",
        "classification",
        "beta_value_abar",
        "beta_value_formula",
    ]
    cols = [c for c in cols if c in selected.columns]

    selected = selected[cols]
    selected.to_csv(FIG_DIR / "representative_galaxies_for_paper.csv", index=False, encoding="utf-8-sig")


# =========================================================
# Table-style summary text 저장
# =========================================================
def save_summary_text(merged: pd.DataFrame, class_df: pd.DataFrame):
    n_total = len(merged)

    improved = int(class_df.loc[class_df["classification"] == "improved", "count"].iloc[0])
    worsened = int(class_df.loc[class_df["classification"] == "worsened", "count"].iloc[0])
    similar = int(class_df.loc[class_df["classification"] == "similar", "count"].iloc[0])

    text = []
    text.append(f"Total galaxies: {n_total}")
    text.append(f"Improved: {improved} ({100*improved/n_total:.1f}%)")
    text.append(f"Worsened: {worsened} ({100*worsened/n_total:.1f}%)")
    text.append(f"Similar: {similar} ({100*similar/n_total:.1f}%)")
    text.append("")
    text.append(f"Mean chi2 (abar): {merged['chi2_abar'].mean():.2f}")
    text.append(f"Mean chi2 (formula): {merged['chi2_formula'].mean():.2f}")
    text.append(f"Median chi2 (abar): {merged['chi2_abar'].median():.2f}")
    text.append(f"Median chi2 (formula): {merged['chi2_formula'].median():.2f}")
    text.append(f"Mean delta chi2: {merged['delta_chi2'].mean():.2f}")

    out_path = FIG_DIR / "paper_figure_summary.txt"
    out_path.write_text("\n".join(text), encoding="utf-8")


# =========================================================
# main
# =========================================================
def main():
    merged, class_df, rep_df = load_data()
    plot_chi2_distribution(merged)
    plot_classification_counts(class_df)
    save_representative_summary(merged)
    save_summary_text(merged, class_df)
    print("논문용 figure 데이터 생성 완료")


if __name__ == "__main__":
    main()