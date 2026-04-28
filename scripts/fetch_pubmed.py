import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "pubmed"
LOCKED_DISEASES_PATH = ROOT / "outputs" / "locked_diseases.csv"
OUTPUT_PATH = OUTPUT_DIR / "abstracts.jsonl"

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
load_dotenv(ROOT / ".env")
TOOL_NAME = os.getenv("NCBI_TOOL", "hw7_agentic_rare_dx")
EMAIL = os.getenv("NCBI_EMAIL", "")
API_KEY = os.getenv("NCBI_API_KEY", "")
RETMAX = int(os.getenv("PUBMED_RETMAX", "35"))
SLEEP_SECONDS = float(os.getenv("PUBMED_SLEEP_SECONDS", "0.4"))


def build_common_params() -> dict[str, str]:
    params = {"tool": TOOL_NAME}
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    return params


def build_queries(disease_name: str) -> list[str]:
    return [
        f'"{disease_name}"[Title/Abstract]',
        f'"{disease_name}"[MeSH Terms] OR "{disease_name}"[Title/Abstract]',
        (
            f'"{disease_name}"[Title/Abstract] AND '
            "(diagnosis[Title/Abstract] OR phenotype[Title/Abstract] "
            "OR symptom*[Title/Abstract] OR case[Title/Abstract])"
        ),
        f'"{disease_name}"[All Fields]',
    ]


def esearch(session: requests.Session, term: str) -> list[str]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(RETMAX),
        "retmode": "xml",
        "sort": "relevance",
        **build_common_params(),
    }
    response = session.get(
        f"{BASE_URL}/esearch.fcgi", params=params, timeout=60
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return [node.text for node in root.findall(".//Id") if node.text]


def efetch(
    session: requests.Session, pmids: list[str]
) -> list[dict[str, Any]]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        **build_common_params(),
    }
    response = session.get(
        f"{BASE_URL}/efetch.fcgi", params=params, timeout=60
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    records: list[dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = (
            "".join(article.find(".//ArticleTitle").itertext()).strip()
            if article.find(".//ArticleTitle") is not None
            else ""
        )
        abstract_parts = []
        for abstract_node in article.findall(".//AbstractText"):
            label = abstract_node.attrib.get("Label", "").strip()
            text = "".join(abstract_node.itertext()).strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts).strip()
        journal = article.findtext(".//Journal/Title", default="").strip()
        year = article.findtext(".//PubDate/Year", default="").strip()
        mesh_terms = [
            "".join(node.itertext()).strip()
            for node in article.findall(".//MeshHeading/DescriptorName")
            if "".join(node.itertext()).strip()
        ]
        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "mesh_terms": mesh_terms,
            }
        )
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diseases = (
        pd.read_csv(LOCKED_DISEASES_PATH)["disease_name"]
        .drop_duplicates()
        .tolist()
    )

    session = requests.Session()
    all_records: list[dict[str, Any]] = []

    for disease_name in diseases:
        used_query = ""
        pmids: list[str] = []
        for query in build_queries(disease_name):
            pmids = esearch(session, query)
            time.sleep(SLEEP_SECONDS)
            if pmids:
                used_query = query
                break

        if not pmids:
            print(f"{disease_name}: no PubMed hits")
            continue

        records = efetch(session, pmids)
        time.sleep(SLEEP_SECONDS)

        for record in records:
            record["query_disease"] = disease_name
            record["query"] = used_query
        all_records.extend(records)
        print(f"{disease_name}: {len(records)} abstracts via {used_query}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_records)} abstracts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
