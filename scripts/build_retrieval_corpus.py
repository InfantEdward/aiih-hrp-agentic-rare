import json
import re
from pathlib import Path
import pandas as pd
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCKED_DISEASES_PATH = OUTPUT_DIR / "locked_diseases.csv"
PUBMED_PATH = DATA_DIR / "pubmed" / "abstracts.jsonl"
MEDQUAD_PATH = DATA_DIR / "medquad"
CORPUS_PATH = OUTPUT_DIR / "retrieval_corpus.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "retrieval_corpus_summary.csv"


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set[str]:
    return set(normalize(text).split())


def load_locked_diseases() -> pd.DataFrame:
    locked = pd.read_csv(LOCKED_DISEASES_PATH)
    locked["normalized_name"] = locked["disease_name"].map(normalize)
    locked["name_tokens"] = locked["disease_name"].map(tokenize)
    return locked


def medquad_matches_disease(
    row: dict, disease_name: str, disease_norm: str, disease_tokens: set[str]
) -> bool:
    focus = normalize(row.get("question_focus") or "")
    synonyms = normalize(row.get("synonyms") or "")
    url = normalize(
        (row.get("document_url") or "")
        .rstrip("/")
        .split("/")[-1]
        .replace("-", " ")
    )
    haystacks = [focus, synonyms, url]

    if disease_norm in haystacks:
        return True
    if any(disease_norm in h for h in haystacks if h):
        return True

    for haystack in haystacks:
        tokens = set(haystack.split())
        if disease_tokens and disease_tokens.issubset(tokens):
            return True
    return False


def build_medquad_records(locked: pd.DataFrame) -> list[dict]:
    ds = load_from_disk(str(MEDQUAD_PATH))["train"]
    records: list[dict] = []
    seen_ids: set[str] = set()

    for disease in locked.itertuples(index=False):
        for row in ds:
            if not medquad_matches_disease(
                row,
                disease.disease_name,
                disease.normalized_name,
                disease.name_tokens,
            ):
                continue

            record_id = f"medquad::{row['question_id']}"
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            text = (
                f"Disease: {disease.disease_name}\n"
                f"Question: {row['question']}\n"
                f"Answer: {row['answer']}"
            )
            records.append(
                {
                    "record_id": record_id,
                    "source": "medquad",
                    "disease_name": disease.disease_name,
                    "database_id": disease.database_id,
                    "mondo_id": disease.mondo_id,
                    "title": row["question_focus"] or disease.disease_name,
                    "text": text,
                    "metadata": {
                        "document_source": row.get("document_source"),
                        "document_url": row.get("document_url"),
                        "question_type": row.get("question_type"),
                        "umls_cui": row.get("umls_cui"),
                    },
                }
            )
    return records


def build_pubmed_records(locked_names: set[str]) -> list[dict]:
    records: list[dict] = []
    seen_pmids: set[str] = set()
    with PUBMED_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            disease_name = row["query_disease"]
            if disease_name not in locked_names:
                continue
            pmid = row.get("pmid") or ""
            record_id = f"pubmed::{pmid or len(records)}"
            if pmid and pmid in seen_pmids:
                continue
            if pmid:
                seen_pmids.add(pmid)
            abstract = (row.get("abstract") or "").strip()
            if not abstract:
                continue
            text = f"Disease: {disease_name}\nTitle: {row.get('title','')}\nAbstract: {abstract}"
            records.append(
                {
                    "record_id": record_id,
                    "source": "pubmed",
                    "disease_name": disease_name,
                    "database_id": None,
                    "mondo_id": None,
                    "title": row.get("title", ""),
                    "text": text,
                    "metadata": {
                        "pmid": pmid,
                        "journal": row.get("journal"),
                        "year": row.get("year"),
                        "query": row.get("query"),
                    },
                }
            )
    return records


def main() -> None:
    locked = load_locked_diseases()
    medquad_records = build_medquad_records(locked)
    pubmed_records = build_pubmed_records(set(locked["disease_name"]))
    corpus = medquad_records + pubmed_records

    with CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for record in corpus:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = (
        pd.DataFrame(corpus)
        .groupby(["disease_name", "source"], as_index=False)
        .agg(num_documents=("record_id", "size"))
        .sort_values(["disease_name", "source"])
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved {len(corpus)} retrieval records to {CORPUS_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
