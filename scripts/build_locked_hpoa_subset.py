from pathlib import Path
import pandas as pd
import pronto


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPOA_PATH = DATA_DIR / "hpo" / "phenotype.hpoa"
HPO_PATH = DATA_DIR / "hpo" / "hp.obo"
LOCKED_DISEASES_PATH = OUTPUT_DIR / "locked_diseases.csv"
OUT_ANNOTATIONS = OUTPUT_DIR / "locked_hpoa_annotations.csv"
OUT_SUMMARY = OUTPUT_DIR / "locked_disease_summary.csv"

FREQUENCY_MAP = {
    "HP:0040280": ("Obligate", 1.000),
    "HP:0040281": ("Very frequent", 0.895),
    "HP:0040282": ("Frequent", 0.545),
    "HP:0040283": ("Occasional", 0.170),
    "HP:0040284": ("Very rare", 0.025),
}


def parse_frequency(raw: str) -> tuple[str, float]:
    raw = (raw or "").strip()
    if not raw:
        return "Unspecified", 0.500
    if raw in FREQUENCY_MAP:
        return FREQUENCY_MAP[raw]
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            den_value = float(den)
            if den_value == 0:
                return raw, 0.0
            return raw, max(0.0, min(1.0, float(num) / den_value))
        except ValueError:
            return raw, 0.500
    if raw.endswith("%"):
        try:
            return raw, max(0.0, min(1.0, float(raw[:-1]) / 100.0))
        except ValueError:
            return raw, 0.500
    return raw, 0.500


def main() -> None:
    locked = pd.read_csv(LOCKED_DISEASES_PATH, dtype=str)
    hpoa = pd.read_csv(HPOA_PATH, sep="\t", comment="#", dtype=str).fillna("")
    hpo = pronto.Ontology(str(HPO_PATH))

    hpoa = hpoa[(hpoa["qualifier"] == "") & (hpoa["aspect"] == "P")].copy()
    subset = hpoa[hpoa["database_id"].isin(set(locked["database_id"]))].copy()
    subset = subset.merge(
        locked[["database_id", "disease_name", "mondo_id"]],
        on=["database_id", "disease_name"],
        how="inner",
    )

    parsed = subset["frequency"].map(parse_frequency)
    subset["frequency_label"] = parsed.map(lambda item: item[0])
    subset["frequency_probability"] = parsed.map(lambda item: item[1]).round(3)
    subset["hpo_name"] = subset["hpo_id"].map(
        lambda hpo_id: hpo[hpo_id].name if hpo_id in hpo else hpo_id
    )

    subset = subset.sort_values(
        ["disease_name", "frequency_probability", "hpo_id"],
        ascending=[True, False, True],
    )
    subset.to_csv(OUT_ANNOTATIONS, index=False)

    summary = (
        subset.groupby(
            ["database_id", "disease_name", "mondo_id"], as_index=False
        )
        .agg(
            annotation_rows=("hpo_id", "size"),
            unique_phenotypes=("hpo_id", "nunique"),
            mean_probability=("frequency_probability", "mean"),
            high_conf_phenotypes=(
                "frequency_probability",
                lambda s: int((s >= 0.5).sum()),
            ),
        )
        .sort_values(["unique_phenotypes", "annotation_rows"], ascending=False)
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved {len(subset)} annotation rows to {OUT_ANNOTATIONS}")
    print(f"Saved disease summary to {OUT_SUMMARY}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
