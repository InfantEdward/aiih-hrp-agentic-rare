import json
import math
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path
import chromadb
import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
ENV_VALUES = dotenv_values(ENV_FILE)

if ENV_VALUES.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ENV_VALUES["OPENAI_API_KEY"]

OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

PATIENTS_PATH = OUTPUT_DIR / "simulated_patients.jsonl"
ANNOTATIONS_PATH = OUTPUT_DIR / "locked_hpoa_annotations.csv"
GRAPH_PATH = OUTPUT_DIR / "locked_disease_graph.pkl"
CHROMA_DIR = DATA_DIR / "chroma"

SUITE_PREDICTIONS_PATH = OUTPUT_DIR / "agent_eval_suite_predictions.jsonl"
SUITE_SUMMARY_PATH = OUTPUT_DIR / "agent_eval_suite_summary.csv"
SUITE_METRICS_PATH = OUTPUT_DIR / "agent_eval_suite_metrics.json"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hw7_retrieval_corpus")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MAX_PATIENTS = int(os.getenv("AGENT_EVAL_MAX_PATIENTS", "0"))
PROFILE_PHENOTYPE_LIMIT_FOR_LLM = int(
    os.getenv("AGENT_PROFILE_PHENOTYPE_LIMIT_FOR_LLM", "8")
)
RETRIEVAL_K = int(os.getenv("AGENT_RETRIEVAL_K", "8"))
EVIDENCE_PER_DISEASE = int(os.getenv("AGENT_EVIDENCE_PER_DISEASE", "2"))
PROFILE_PHENOTYPES = int(os.getenv("AGENT_PROFILE_PHENOTYPES", "8"))
HEURISTIC_SIGNATURE_TOP_K = int(os.getenv("HEURISTIC_SIGNATURE_TOP_K", "3"))
GRAPH_SEMANTIC_WEIGHT = float(os.getenv("GRAPH_SEMANTIC_WEIGHT", "0.35"))
GRAPH_MAX_COMMON_DISTANCE = int(os.getenv("GRAPH_MAX_COMMON_DISTANCE", "4"))
GRAPH_EXAMPLE_LIMIT = int(os.getenv("GRAPH_EXAMPLE_LIMIT", "3"))

AVAILABLE_MODES = ("static", "rag", "rag_graph")
RAW_EXPERIMENTS = os.getenv("AGENT_EXPERIMENTS", "all").strip().lower()


def parse_modes(raw: str) -> list[str]:
    if not raw or raw == "all":
        return list(AVAILABLE_MODES)
    modes = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if mode in AVAILABLE_MODES and mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError(
            f"AGENT_EXPERIMENTS must be one of {AVAILABLE_MODES} or 'all'."
        )
    return modes


def load_patients() -> list[dict]:
    patients: list[dict] = []
    with PATIENTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            patients.append(json.loads(line))
    if MAX_PATIENTS > 0:
        return patients[:MAX_PATIENTS]
    return patients


def load_disease_profiles() -> dict[str, dict]:
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    annotations["frequency_probability"] = annotations[
        "frequency_probability"
    ].astype(float)
    num_diseases = annotations["disease_name"].nunique()
    phenotype_disease_counts = annotations.groupby("hpo_id")[
        "disease_name"
    ].nunique()
    phenotype_rarity = {
        hpo_id: math.log((1 + num_diseases) / (1 + count)) + 1.0
        for hpo_id, count in phenotype_disease_counts.items()
    }

    profiles: dict[str, dict] = {}
    for disease_name, rows in annotations.groupby("disease_name", sort=True):
        deduped = rows.sort_values(
            ["frequency_probability", "hpo_id"], ascending=[False, True]
        ).drop_duplicates("hpo_id")
        phenotype_rows = [
            {
                "hpo_id": row.hpo_id,
                "hpo_name": row.hpo_name,
                "frequency_probability": float(row.frequency_probability),
            }
            for row in deduped.itertuples(index=False)
        ]
        phenotype_weights = {
            item["hpo_id"]: item["frequency_probability"]
            for item in phenotype_rows
        }
        phenotype_rarity_weights = {
            item["hpo_id"]: phenotype_rarity.get(item["hpo_id"], 1.0)
            for item in phenotype_rows
        }
        profiles[disease_name] = {
            "disease_name": disease_name,
            "database_id": rows["database_id"].iloc[0],
            "mondo_id": rows["mondo_id"].iloc[0],
            "phenotype_rows": phenotype_rows,
            "phenotype_ids": set(phenotype_weights),
            "phenotype_weights": phenotype_weights,
            "phenotype_rarity_weights": phenotype_rarity_weights,
            "profile_summary": [
                {
                    "hpo_id": item["hpo_id"],
                    "hpo_name": item["hpo_name"],
                    "frequency_probability": round(
                        item["frequency_probability"], 3
                    ),
                }
                for item in phenotype_rows[:PROFILE_PHENOTYPES]
            ],
        }
    return profiles


def load_graph():
    with GRAPH_PATH.open("rb") as handle:
        return pickle.load(handle)


def compute_ancestor_distances(
    graph, phenotype_ids: set[str]
) -> dict[str, dict[str, int]]:
    distances: dict[str, dict[str, int]] = {}
    for phenotype_id in phenotype_ids:
        if phenotype_id not in graph:
            continue
        seen = {phenotype_id: 0}
        queue = deque([phenotype_id])
        while queue:
            current = queue.popleft()
            current_distance = seen[current]
            for _, parent, edge_data in graph.out_edges(current, data=True):
                if edge_data.get("relation") != "is_a":
                    continue
                next_distance = current_distance + 1
                if parent not in seen or next_distance < seen[parent]:
                    seen[parent] = next_distance
                    queue.append(parent)
        distances[phenotype_id] = seen
    return distances


def score_candidates_overlap(
    patient: dict, profiles: dict[str, dict]
) -> list[dict]:
    observed_ids = {item["hpo_id"] for item in patient["all_phenotypes"]}
    scored: list[dict] = []

    for disease_name, profile in profiles.items():
        rarity_weights = profile["phenotype_rarity_weights"]
        profile_rows = profile["phenotype_rows"]
        profile_ids = profile["phenotype_ids"]
        top_signature_ids = {
            item["hpo_id"] for item in profile_rows[:HEURISTIC_SIGNATURE_TOP_K]
        }

        exact_match_items = [
            {
                "hpo_id": item["hpo_id"],
                "hpo_name": item["hpo_name"],
                "frequency_probability": round(
                    item["frequency_probability"], 3
                ),
                "rarity_weight": round(
                    rarity_weights.get(item["hpo_id"], 1.0), 3
                ),
            }
            for item in profile_rows
            if item["hpo_id"] in observed_ids
        ]
        match_score = sum(
            rarity_weights.get(item["hpo_id"], 1.0)
            for item in exact_match_items
        )
        signature_bonus = sum(
            0.2
            for item in exact_match_items
            if item["hpo_id"] in top_signature_ids
        )
        exact_matches = len(exact_match_items)
        coverage = exact_matches / max(1, len(observed_ids))
        normalized_score = (match_score + signature_bonus) / math.sqrt(
            max(1, len(profile_ids))
        )
        scored.append(
            {
                "disease_name": disease_name,
                "database_id": profile["database_id"],
                "mondo_id": profile["mondo_id"],
                "match_score": round(match_score + signature_bonus, 4),
                "normalized_score": round(normalized_score, 4),
                "exact_matches": int(exact_matches),
                "coverage": round(coverage, 4),
                "exact_match_items": exact_match_items[:PROFILE_PHENOTYPES],
                "profile_summary": profile["profile_summary"],
            }
        )

    scored.sort(
        key=lambda item: (
            item["normalized_score"],
            item["match_score"],
            item["exact_matches"],
        ),
        reverse=True,
    )
    return scored


def score_candidates_graph(
    patient: dict,
    profiles: dict[str, dict],
    graph,
    ancestor_distances: dict[str, dict[str, int]],
) -> list[dict]:
    observed_rows = patient["all_phenotypes"]
    observed_ids = {item["hpo_id"] for item in observed_rows}
    observed_name_lookup = {
        item["hpo_id"]: item["hpo_name"] for item in observed_rows
    }

    scored: list[dict] = []
    for disease_name, profile in profiles.items():
        rarity_weights = profile["phenotype_rarity_weights"]
        exact_score = sum(
            rarity_weights.get(hpo_id, 1.0)
            for hpo_id in observed_ids
            if hpo_id in profile["phenotype_ids"]
        )
        exact_matches = [
            {
                "hpo_id": item["hpo_id"],
                "hpo_name": item["hpo_name"],
                "frequency_probability": round(
                    item["frequency_probability"], 3
                ),
                "rarity_weight": round(
                    rarity_weights.get(item["hpo_id"], 1.0), 3
                ),
            }
            for item in profile["phenotype_rows"]
            if item["hpo_id"] in observed_ids
        ]

        semantic_score = 0.0
        semantic_examples = []
        for disease_item in profile["phenotype_rows"]:
            disease_hpo_id = disease_item["hpo_id"]
            if disease_hpo_id in observed_ids:
                continue

            disease_ancestors = ancestor_distances.get(
                disease_hpo_id, {disease_hpo_id: 0}
            )
            best = None
            for observed_item in observed_rows:
                observed_hpo_id = observed_item["hpo_id"]
                observed_ancestors = ancestor_distances.get(
                    observed_hpo_id, {observed_hpo_id: 0}
                )
                common = set(observed_ancestors).intersection(
                    disease_ancestors
                )
                if not common:
                    continue

                best_for_pair = None
                for ancestor_id in common:
                    total_distance = (
                        observed_ancestors[ancestor_id]
                        + disease_ancestors[ancestor_id]
                    )
                    if total_distance > GRAPH_MAX_COMMON_DISTANCE:
                        continue
                    closeness = 1.0 / (1.0 + total_distance)
                    if (
                        best_for_pair is None
                        or closeness > best_for_pair["closeness"]
                    ):
                        best_for_pair = {
                            "ancestor_id": ancestor_id,
                            "ancestor_name": graph.nodes[ancestor_id].get(
                                "name", ancestor_id
                            ),
                            "total_distance": total_distance,
                            "closeness": closeness,
                            "observed_hpo_id": observed_hpo_id,
                            "observed_hpo_name": observed_name_lookup.get(
                                observed_hpo_id, observed_hpo_id
                            ),
                        }
                if best_for_pair is None:
                    continue
                if (
                    best is None
                    or best_for_pair["closeness"] > best["closeness"]
                ):
                    best = best_for_pair

            if best is not None:
                weighted = (
                    rarity_weights.get(disease_hpo_id, 1.0)
                    * GRAPH_SEMANTIC_WEIGHT
                    * best["closeness"]
                )
                semantic_score += weighted
                semantic_examples.append(
                    {
                        "observed_hpo_name": best["observed_hpo_name"],
                        "disease_hpo_name": disease_item["hpo_name"],
                        "bridge_ancestor_name": best["ancestor_name"],
                        "distance": best["total_distance"],
                        "closeness": round(best["closeness"], 3),
                    }
                )

        semantic_examples.sort(
            key=lambda item: (item["closeness"], -item["distance"]),
            reverse=True,
        )
        total_score = exact_score + semantic_score
        normalized_graph_score = total_score / math.sqrt(
            max(1, len(profile["phenotype_ids"]))
        )
        scored.append(
            {
                "disease_name": disease_name,
                "database_id": profile["database_id"],
                "mondo_id": profile["mondo_id"],
                "graph_score": round(total_score, 4),
                "normalized_graph_score": round(normalized_graph_score, 4),
                "exact_match_count": len(exact_matches),
                "semantic_match_count": len(semantic_examples),
                "exact_matches": exact_matches[:PROFILE_PHENOTYPES],
                "semantic_examples": semantic_examples[:GRAPH_EXAMPLE_LIMIT],
                "profile_summary": profile["profile_summary"],
            }
        )

    scored.sort(
        key=lambda item: (
            item["normalized_graph_score"],
            item["graph_score"],
            item["exact_match_count"],
            item["semantic_match_count"],
        ),
        reverse=True,
    )
    return scored


def build_query_text(patient: dict) -> str:
    phenotype_lines = [
        f"- {item['hpo_name']} ({item['hpo_id']})"
        for item in patient["all_phenotypes"]
    ]
    return "\n".join(
        [
            f"Rare disease differential diagnosis for patient {patient['patient_id']}.",
            "Observed phenotypes:",
            *phenotype_lines,
        ]
    )


def retrieve_evidence(
    chroma_collection,
    client: OpenAI,
    patient: dict,
    allowed_diseases: set[str],
) -> dict[str, list[dict]]:
    query_text = build_query_text(patient)
    query_embedding = (
        client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
        .data[0]
        .embedding
    )

    raw = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=max(RETRIEVAL_K * 3, RETRIEVAL_K),
        include=["documents", "metadatas", "distances"],
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        disease_name = (metadata or {}).get("disease_name")
        if disease_name not in allowed_diseases:
            continue
        if len(grouped[disease_name]) >= EVIDENCE_PER_DISEASE:
            continue
        grouped[disease_name].append(
            {
                "title": (metadata or {}).get("title", ""),
                "source": (metadata or {}).get("source", ""),
                "distance": float(distance) if distance is not None else None,
                "document": document,
            }
        )

    return dict(grouped)


def build_full_candidate_list(
    profiles: dict[str, dict],
    mode: str,
    graph_scores_by_disease: dict[str, dict] | None,
) -> list[dict]:
    """Return one minimal entry per disease in the locked set, alphabetised so
    no ordering can leak the answer. Score fields and exact-match counts are
    deliberately omitted — the LLM has to do the work. For rag_graph mode,
    we attach graph-derived ancestor bridges (semantic_examples) per disease
    as qualitative hints, but no numeric score."""
    candidates = []
    for disease_name in sorted(profiles):
        profile = profiles[disease_name]
        entry = {
            "disease_name": disease_name,
            "database_id": profile["database_id"],
            "mondo_id": profile["mondo_id"],
            "profile_summary": profile["profile_summary"],
        }
        if mode == "rag_graph" and graph_scores_by_disease is not None:
            graph_item = graph_scores_by_disease.get(disease_name)
            if graph_item is not None and graph_item["semantic_examples"]:
                entry["ontology_bridges"] = graph_item["semantic_examples"]
        candidates.append(entry)
    return candidates


def heuristic_ranking(overlap_scores: list[dict]) -> list[str]:
    return [item["disease_name"] for item in overlap_scores]


def build_prompt(
    mode: str,
    patient: dict,
    candidates: list[dict],
    evidence_by_disease: dict[str, list[dict]],
) -> str:
    task = (
        "You are given a patient's observed phenotype list and the FULL catalogue of "
        "candidate rare diseases (alphabetised, no scores). Decide the most likely "
        "diagnosis. Note that some observed phenotypes may be generalised (parent terms "
        "instead of leaf terms) or may be confounders carried over from related "
        "diseases. Return the three most likely diseases, ranked."
    )
    if mode == "rag":
        task = (
            "You are given a patient's observed phenotype list, the FULL catalogue of "
            "candidate rare diseases (alphabetised, no scores), and retrieved text "
            "evidence per disease (only diseases with retrieved snippets are populated). "
            "Some observed phenotypes may be generalised parent terms or confounders "
            "from related diseases. Use the text evidence to disambiguate. Return the "
            "three most likely diseases, ranked."
        )
    elif mode == "rag_graph":
        task = (
            "You are given a patient's observed phenotype list, the FULL catalogue of "
            "candidate rare diseases (alphabetised, no scores), retrieved text evidence, "
            "and ontology-bridge hints (`ontology_bridges`) showing how observed phenotypes "
            "share an HPO ancestor with that disease's known phenotypes. Some observed "
            "phenotypes may be generalised parent terms or confounders from related "
            "diseases — the bridges are designed to recover those matches. Return the "
            "three most likely diseases, ranked."
        )

    payload = {
        "experiment_mode": mode,
        "patient_id": patient["patient_id"],
        "observed_phenotypes": patient["all_phenotypes"],
        "candidate_diseases": candidates,
        "retrieved_evidence": evidence_by_disease,
        "task": task,
        "response_schema": {
            "ranked_diseases": [
                {
                    "disease_name": "string",
                    "reason": "string",
                }
            ],
            "final_diagnosis": "string",
            "confidence": "number between 0 and 1",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def rerank_with_llm(
    client: OpenAI,
    mode: str,
    patient: dict,
    candidates: list[dict],
    evidence_by_disease: dict[str, list[dict]],
) -> dict:
    candidate_names = {item["disease_name"] for item in candidates}
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical-research assistant for rare disease "
                    "ranking. Do not invent candidate diseases. Only rank from the "
                    "provided candidate list. Return a valid JSON object only."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
                    mode, patient, candidates, evidence_by_disease
                ),
            },
        ],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")

    ranked = []
    for item in parsed.get("ranked_diseases", []):
        disease_name = item.get("disease_name", "").strip()
        if disease_name in candidate_names and disease_name not in {
            x["disease_name"] for x in ranked
        }:
            ranked.append(
                {
                    "disease_name": disease_name,
                    "reason": item.get("reason", "").strip(),
                }
            )

    for candidate in candidates:
        if len(ranked) >= 3:
            break
        if candidate["disease_name"] not in {
            x["disease_name"] for x in ranked
        }:
            ranked.append(
                {
                    "disease_name": candidate["disease_name"],
                    "reason": "Backfilled from heuristic shortlist.",
                }
            )

    final_diagnosis = parsed.get("final_diagnosis", "").strip()
    if final_diagnosis not in candidate_names:
        final_diagnosis = (
            ranked[0]["disease_name"]
            if ranked
            else candidates[0]["disease_name"]
        )

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "final_diagnosis": final_diagnosis,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "ranked_diseases": ranked[:3],
        "raw_response": parsed,
    }


def reciprocal_rank(ranking: list[str], target: str) -> float:
    for index, name in enumerate(ranking, start=1):
        if name == target:
            return 1.0 / index
    return 0.0


def compute_mode_metrics(records: list[dict], mode: str) -> dict:
    mode_rows = [row for row in records if row["mode"] == mode]
    num_patients = len(mode_rows)
    if num_patients == 0:
        return {"num_patients": 0}

    heuristic_top1 = sum(
        1
        for row in mode_rows
        if row["heuristic_ranked_diseases"][:1] == [row["true_diagnosis"]]
    )
    heuristic_top3 = sum(
        1
        for row in mode_rows
        if row["true_diagnosis"] in row["heuristic_ranked_diseases"][:3]
    )
    heuristic_mrr = (
        sum(
            reciprocal_rank(
                row["heuristic_ranked_diseases"], row["true_diagnosis"]
            )
            for row in mode_rows
        )
        / num_patients
    )

    llm_top1 = sum(
        1
        for row in mode_rows
        if row["llm_ranked_diseases"][:1] == [row["true_diagnosis"]]
    )
    llm_top3 = sum(
        1
        for row in mode_rows
        if row["true_diagnosis"] in row["llm_ranked_diseases"][:3]
    )
    llm_mrr = (
        sum(
            reciprocal_rank(row["llm_ranked_diseases"], row["true_diagnosis"])
            for row in mode_rows
        )
        / num_patients
    )

    return {
        "num_patients": num_patients,
        "heuristic_top1_accuracy": round(heuristic_top1 / num_patients, 4),
        "heuristic_top3_accuracy": round(heuristic_top3 / num_patients, 4),
        "heuristic_mrr": round(heuristic_mrr, 4),
        "llm_top1_accuracy": round(llm_top1 / num_patients, 4),
        "llm_top3_accuracy": round(llm_top3 / num_patients, 4),
        "llm_mrr": round(llm_mrr, 4),
        "delta_top1_llm_minus_heuristic": round(
            (llm_top1 - heuristic_top1) / num_patients, 4
        ),
        "delta_top3_llm_minus_heuristic": round(
            (llm_top3 - heuristic_top3) / num_patients, 4
        ),
        "delta_mrr_llm_minus_heuristic": round(llm_mrr - heuristic_mrr, 4),
        "mean_candidate_count": round(
            sum(row["num_candidates"] for row in mode_rows) / num_patients, 3
        ),
        "mean_retrieved_docs": round(
            sum(row["num_retrieved_docs"] for row in mode_rows) / num_patients,
            3,
        ),
        "mean_confidence": round(
            sum(row["confidence"] for row in mode_rows) / num_patients, 4
        ),
    }


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    modes = parse_modes(RAW_EXPERIMENTS)
    patients = load_patients()
    profiles = load_disease_profiles()
    graph = load_graph() if "rag_graph" in modes else None

    phenotype_ids = {
        item["hpo_id"]
        for profile in profiles.values()
        for item in profile["phenotype_rows"]
    }
    phenotype_ids.update(
        item["hpo_id"]
        for patient in patients
        for item in patient["all_phenotypes"]
    )
    ancestor_distances = (
        compute_ancestor_distances(graph, phenotype_ids)
        if graph is not None
        else {}
    )

    client = OpenAI()
    collection = None
    if any(mode in {"rag", "rag_graph"} for mode in modes):
        if not CHROMA_DIR.exists():
            raise RuntimeError(f"Chroma directory not found: {CHROMA_DIR}")
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

    prediction_rows: list[dict] = []

    all_disease_names = set(profiles)

    for patient_index, patient in enumerate(patients, start=1):
        overlap_scores = score_candidates_overlap(patient, profiles)
        heuristic_ranking_full = heuristic_ranking(overlap_scores)
        graph_scores = (
            score_candidates_graph(
                patient, profiles, graph, ancestor_distances
            )
            if graph is not None
            else None
        )
        graph_scores_by_disease = (
            {item["disease_name"]: item for item in graph_scores}
            if graph_scores is not None
            else None
        )

        shared_evidence = {}
        if (
            any(mode in {"rag", "rag_graph"} for mode in modes)
            and collection is not None
        ):
            shared_evidence = retrieve_evidence(
                collection, client, patient, all_disease_names
            )

        for mode in modes:
            candidates = build_full_candidate_list(
                profiles, mode, graph_scores_by_disease
            )
            candidate_names = [item["disease_name"] for item in candidates]

            evidence_by_disease: dict[str, list[dict]] = {}
            if mode in {"rag", "rag_graph"}:
                evidence_by_disease = {
                    disease_name: shared_evidence[disease_name]
                    for disease_name in candidate_names
                    if disease_name in shared_evidence
                }

            llm_result = rerank_with_llm(
                client, mode, patient, candidates, evidence_by_disease
            )

            row = {
                "mode": mode,
                "patient_id": patient["patient_id"],
                "true_diagnosis": patient["true_diagnosis"],
                "database_id": patient["database_id"],
                "mondo_id": patient["mondo_id"],
                "num_phenotypes": len(patient["all_phenotypes"]),
                "num_candidates": len(candidates),
                "heuristic_ranked_diseases": heuristic_ranking_full,
                "llm_ranked_diseases": [
                    item["disease_name"]
                    for item in llm_result["ranked_diseases"]
                ],
                "predicted_diagnosis": llm_result["final_diagnosis"],
                "confidence": llm_result["confidence"],
                "retrieved_evidence": evidence_by_disease,
                "llm_reasons": llm_result["ranked_diseases"],
                "num_retrieved_docs": sum(
                    len(items) for items in evidence_by_disease.values()
                ),
            }
            prediction_rows.append(row)
            print(
                f"[{patient_index}/{len(patients)}][{mode}] {row['patient_id']}: "
                f"true={row['true_diagnosis']} pred={row['predicted_diagnosis']}"
            )

    with SUITE_PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = [
        {
            "mode": row["mode"],
            "patient_id": row["patient_id"],
            "true_diagnosis": row["true_diagnosis"],
            "predicted_diagnosis": row["predicted_diagnosis"],
            "heuristic_top1_correct": row["heuristic_ranked_diseases"][0]
            == row["true_diagnosis"],
            "heuristic_top3_correct": row["true_diagnosis"]
            in row["heuristic_ranked_diseases"][:3],
            "llm_top1_correct": row["llm_ranked_diseases"][0]
            == row["true_diagnosis"],
            "llm_top3_correct": row["true_diagnosis"]
            in row["llm_ranked_diseases"][:3],
            "num_retrieved_docs": row["num_retrieved_docs"],
            "confidence": row["confidence"],
        }
        for row in prediction_rows
    ]
    pd.DataFrame(summary_rows).to_csv(SUITE_SUMMARY_PATH, index=False)

    metrics = {
        "config": {
            "modes": modes,
            "num_patients": len(patients),
            "candidate_count_per_patient": len(profiles),
            "retrieval_k": RETRIEVAL_K,
            "evidence_per_disease": EVIDENCE_PER_DISEASE,
            "chat_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        },
        "metrics_by_mode": {
            mode: compute_mode_metrics(prediction_rows, mode) for mode in modes
        },
    }
    SUITE_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved suite predictions to {SUITE_PREDICTIONS_PATH}")
    print(f"Saved suite summary to {SUITE_SUMMARY_PATH}")
    print(f"Saved suite metrics to {SUITE_METRICS_PATH}")


if __name__ == "__main__":
    main()
