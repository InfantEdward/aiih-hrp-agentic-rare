import math
import os
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import pronto
from datasets import load_from_disk
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
ENV_VALUES = dotenv_values(ROOT / ".env")
for key, value in ENV_VALUES.items():
    if value is not None and key not in os.environ:
        os.environ[key] = value

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPOA_PATH = DATA_DIR / "hpo" / "phenotype.hpoa"
MONDO_PATH = DATA_DIR / "mondo" / "mondo.obo"
MEDQUAD_PATH = DATA_DIR / "medquad"
LOCKED_DISEASE_COUNT = int(os.getenv("LOCKED_DISEASE_COUNT", "12"))
LOCKED_DISEASE_NAMES = [
    item.strip()
    for item in os.getenv("LOCKED_DISEASE_NAMES", "").split(",")
    if item.strip()
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace(",", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_aliases(raw: str | None) -> list[str]:
    if not raw:
        return []
    aliases = [raw]
    aliases.extend(re.split(r"[;|,/]", raw))
    return [a.strip() for a in aliases if a.strip()]


def normalize_source_id(source_id: str) -> str:
    if source_id.startswith("ORPHA:"):
        return source_id.replace("ORPHA:", "ORPHANET:")
    return source_id


def load_hpoa() -> pd.DataFrame:
    df = pd.read_csv(HPOA_PATH, sep="\t", comment="#", dtype=str).fillna("")
    # Keep positive phenotype assertions only; inheritance/onset rows are not useful
    # for phenotype-profile simulation.
    df = df[(df["qualifier"] == "") & (df["aspect"] == "P")].copy()
    return df


def build_mondo_xref_map() -> dict[str, str]:
    mondo = pronto.Ontology(str(MONDO_PATH))
    xref_to_mondo: dict[str, str] = {}
    for term in mondo.terms():
        for xref in term.xrefs:
            xref_to_mondo[str(xref.id).upper()] = term.id
    return xref_to_mondo


def build_medquad_alias_counter() -> Counter[str]:
    ds = load_from_disk(str(MEDQUAD_PATH))["train"]
    counter: Counter[str] = Counter()
    for row in ds:
        aliases = []
        aliases.extend(split_aliases(row.get("question_focus")))
        aliases.extend(split_aliases(row.get("synonyms")))
        url = row.get("document_url") or ""
        slug = url.rstrip("/").split("/")[-1].replace("-", " ")
        aliases.append(slug)
        for alias in aliases:
            normalized = normalize(alias)
            if normalized:
                counter[normalized] += 1
    return counter


def score_row(row: pd.Series) -> float:
    return (
        math.log1p(row["unique_phenotypes"]) * 3.0
        + math.log1p(row["frequency_rows"]) * 2.0
        + math.log1p(row["medquad_hits"]) * 4.0
        + (2.0 if row["mondo_id"] else 0.0)
    )


def main() -> None:
    hpoa = load_hpoa()
    xref_to_mondo = build_mondo_xref_map()
    medquad_alias_counter = build_medquad_alias_counter()

    grouped = (
        hpoa.groupby(["database_id", "disease_name"], as_index=False)
        .agg(
            annotation_rows=("hpo_id", "size"),
            unique_phenotypes=("hpo_id", "nunique"),
            frequency_rows=("frequency", lambda s: int((s != "").sum())),
            references=("reference", "nunique"),
        )
        .sort_values(["annotation_rows", "unique_phenotypes"], ascending=False)
    )

    grouped["normalized_name"] = grouped["disease_name"].map(normalize)
    grouped["mondo_id"] = grouped["database_id"].map(
        lambda x: xref_to_mondo.get(normalize_source_id(x.upper()), "")
    )
    grouped["medquad_hits"] = grouped["normalized_name"].map(
        lambda name: medquad_alias_counter.get(name, 0)
    )
    grouped["frequency_ratio"] = (
        grouped["frequency_rows"] / grouped["annotation_rows"]
    ).round(3)
    grouped["score"] = grouped.apply(score_row, axis=1).round(3)

    candidate_pool = grouped[
        (grouped["mondo_id"] != "")
        & (grouped["unique_phenotypes"] >= 8)
        & (grouped["frequency_rows"] >= 1)
        & (grouped["medquad_hits"] >= 1)
    ].copy()

    default_seed_names = [
        "Cystic fibrosis",
        "Duchenne muscular dystrophy",
        "Fabry disease",
        "Friedreich ataxia",
        "Gaucher disease",
        "Huntington disease",
        "Marfan syndrome",
        "Neurofibromatosis type 1",
        "Noonan syndrome",
        "Phenylketonuria",
        "Tuberous sclerosis complex",
        "Wilson disease",
    ]
    seed_names = LOCKED_DISEASE_NAMES or default_seed_names

    recommended = candidate_pool[
        candidate_pool["disease_name"].isin(seed_names)
    ].copy()
    recommended = recommended.sort_values(
        ["score", "unique_phenotypes", "frequency_rows"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["mondo_id"])

    if len(recommended) < LOCKED_DISEASE_COUNT:
        backfill = (
            candidate_pool[
                ~candidate_pool["disease_name"].isin(
                    recommended["disease_name"]
                )
            ]
            .sort_values(
                [
                    "medquad_hits",
                    "unique_phenotypes",
                    "frequency_rows",
                    "score",
                ],
                ascending=[False, False, False, False],
            )
            .drop_duplicates(subset=["mondo_id"])
        )
        replacements = backfill.head(
            max(0, LOCKED_DISEASE_COUNT - len(recommended))
        )
        recommended = pd.concat([recommended, replacements], ignore_index=True)

    recommended = recommended.sort_values(
        ["medquad_hits", "unique_phenotypes", "frequency_rows", "score"],
        ascending=[False, False, False, False],
    ).head(LOCKED_DISEASE_COUNT)

    grouped.sort_values(
        ["score", "medquad_hits", "unique_phenotypes", "frequency_rows"],
        ascending=[False, False, False, False],
    ).to_csv(OUTPUT_DIR / "disease_ranking.csv", index=False)
    candidate_pool.sort_values(
        ["medquad_hits", "unique_phenotypes", "frequency_rows", "score"],
        ascending=[False, False, False, False],
    ).to_csv(OUTPUT_DIR / "candidate_pool.csv", index=False)
    recommended.to_csv(OUTPUT_DIR / "locked_diseases.csv", index=False)

    print("Top 20 candidates")
    print(
        candidate_pool.sort_values(
            ["medquad_hits", "unique_phenotypes", "frequency_rows", "score"],
            ascending=[False, False, False, False],
        )[
            [
                "database_id",
                "disease_name",
                "unique_phenotypes",
                "frequency_rows",
                "medquad_hits",
                "mondo_id",
                "score",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print("\nRecommended locked list")
    print(
        recommended[
            [
                "database_id",
                "disease_name",
                "unique_phenotypes",
                "frequency_rows",
                "medquad_hits",
                "mondo_id",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
