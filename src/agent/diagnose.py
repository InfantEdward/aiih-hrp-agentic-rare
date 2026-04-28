from __future__ import annotations
import json
import math
import os
import pickle
import re
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal
import chromadb
import pandas as pd
from dotenv import dotenv_values
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
ENV_VALUES = dotenv_values(ENV_FILE)

if ENV_VALUES.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ENV_VALUES["OPENAI_API_KEY"]

for _k, _v in ENV_VALUES.items():
    if _v is not None and _k not in os.environ:
        os.environ[_k] = _v

OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

PATIENTS_PATH = OUTPUT_DIR / "simulated_patients.jsonl"
ANNOTATIONS_PATH = OUTPUT_DIR / "locked_hpoa_annotations.csv"
GRAPH_PATH = OUTPUT_DIR / "locked_disease_graph.pkl"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hw7_retrieval_corpus")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

RETRIEVAL_K = int(os.getenv("AGENT_RETRIEVAL_K", "8"))
EVIDENCE_PER_DISEASE = int(os.getenv("AGENT_EVIDENCE_PER_DISEASE", "2"))
PROFILE_PHENOTYPES = int(os.getenv("AGENT_PROFILE_PHENOTYPES", "8"))
HEURISTIC_SIGNATURE_TOP_K = int(os.getenv("HEURISTIC_SIGNATURE_TOP_K", "3"))
GRAPH_SEMANTIC_WEIGHT = float(os.getenv("GRAPH_SEMANTIC_WEIGHT", "0.35"))
GRAPH_MAX_COMMON_DISTANCE = int(os.getenv("GRAPH_MAX_COMMON_DISTANCE", "4"))
GRAPH_EXAMPLE_LIMIT = int(os.getenv("GRAPH_EXAMPLE_LIMIT", "3"))
AGENTIC_TOOL_BUDGET = int(os.getenv("AGENTIC_TOOL_BUDGET", "6"))
AGENTIC_RECURSION_LIMIT = int(
    os.getenv("AGENTIC_RECURSION_LIMIT", str((AGENTIC_TOOL_BUDGET * 2) + 4))
)
AGENTIC_OVERLAP_TOP_K = int(os.getenv("AGENTIC_OVERLAP_TOP_K", "8"))
AGENTIC_PROFILE_TOP_PHENOTYPES = int(
    os.getenv("AGENTIC_PROFILE_TOP_PHENOTYPES", "8")
)
AGENTIC_TEXT_EVIDENCE_CHAR_LIMIT = int(
    os.getenv("AGENTIC_TEXT_EVIDENCE_CHAR_LIMIT", "500")
)

Mode = Literal["static", "rag", "rag_graph", "agentic"]
ALL_MODES: tuple[Mode, ...] = ("static", "rag", "rag_graph", "agentic")


class RankedDisease(BaseModel):
    disease_name: str
    reason: str


class AgenticDiagnosis(BaseModel):
    final_diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)
    ranked_diseases: list[RankedDisease] = Field(min_length=3, max_length=3)


class AgenticState(MessagesState):
    structured_response: dict[str, Any]


@dataclass
class TraceEvent:
    step: str
    label: str
    tool: str
    started_at: float
    duration_ms: float = 0.0
    input_summary: dict = field(default_factory=dict)
    output_summary: dict = field(default_factory=dict)


class Trace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    @contextmanager
    def step(
        self,
        name: str,
        label: str,
        tool: str,
        input_summary: dict | None = None,
    ):
        event = TraceEvent(
            step=name,
            label=label,
            tool=tool,
            started_at=time.time(),
            input_summary=dict(input_summary or {}),
        )
        try:
            yield event
        finally:
            event.duration_ms = round(
                (time.time() - event.started_at) * 1000, 1
            )
            self.events.append(event)

    def to_list(self) -> list[dict]:
        return [
            {
                "step": event.step,
                "label": event.label,
                "tool": event.tool,
                "duration_ms": event.duration_ms,
                "input_summary": event.input_summary,
                "output_summary": event.output_summary,
            }
            for event in self.events
        ]


def load_patients() -> list[dict]:
    patients: list[dict] = []
    if not PATIENTS_PATH.exists():
        return patients
    with PATIENTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            patients.append(json.loads(line))
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


def build_phenotype_name_index(graph) -> dict[str, str]:
    name_index: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "phenotype":
            continue
        name = data.get("name")
        if not name:
            continue
        name_index[name.lower()] = node_id
    return name_index


def precompute_ancestor_distances(
    graph, profiles: dict[str, dict]
) -> dict[str, dict[str, int]]:
    phenotype_ids = {
        item["hpo_id"]
        for profile in profiles.values()
        for item in profile["phenotype_rows"]
    }
    return compute_ancestor_distances(graph, phenotype_ids)


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


_TOKEN_BOUNDARY = re.compile(r"[A-Za-z0-9]+")


def _tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_BOUNDARY.findall(text)]


def build_phenotype_candidate_shortlist(
    text: str,
    name_index: dict[str, str],
    graph,
    *,
    max_candidates: int = 30,
) -> list[dict]:
    query = (text or "").strip().lower()
    if not query:
        return []

    query_tokens = set(_tokenize_text(query))
    if not query_tokens:
        return []

    scored: list[dict] = []
    for name, hpo_id in name_index.items():
        name_tokens = set(_tokenize_text(name))
        if not name_tokens:
            continue
        shared = query_tokens.intersection(name_tokens)
        if not shared:
            continue

        token_recall = len(shared) / len(name_tokens)
        token_precision = len(shared) / len(query_tokens)
        fuzzy = SequenceMatcher(None, query, name).ratio()
        score = (
            (0.65 * token_recall) + (0.2 * token_precision) + (0.15 * fuzzy)
        )
        scored.append(
            {
                "hpo_id": hpo_id,
                "hpo_name": graph.nodes[hpo_id].get("name", name),
                "shortlist_score": round(score, 4),
                "shared_tokens": sorted(shared),
            }
        )

    scored.sort(
        key=lambda item: (
            item["shortlist_score"],
            len(item["shared_tokens"]),
            len(item["hpo_name"]),
        ),
        reverse=True,
    )

    deduped: list[dict] = []
    seen_ids: set[str] = set()
    for item in scored:
        if item["hpo_id"] in seen_ids:
            continue
        seen_ids.add(item["hpo_id"])
        deduped.append(item)
        if len(deduped) >= max_candidates:
            break
    return deduped


def extract_phenotypes_from_text(
    text: str, name_index: dict[str, str], graph, max_results: int = 25
) -> list[dict]:
    if not text or not text.strip():
        return []

    masked = text.lower()
    sorted_names = sorted(name_index.keys(), key=len, reverse=True)
    matches: list[dict] = []
    seen_ids: set[str] = set()

    for name in sorted_names:
        if len(name) < 5:
            continue
        if len(matches) >= max_results:
            break
        idx = 0
        while True:
            found = masked.find(name, idx)
            if found < 0:
                break
            # word-boundary check
            before_ok = found == 0 or not masked[found - 1].isalnum()
            end = found + len(name)
            after_ok = end >= len(masked) or not masked[end].isalnum()
            if before_ok and after_ok:
                hpo_id = name_index[name]
                if hpo_id in seen_ids:
                    masked = masked[:found] + (" " * len(name)) + masked[end:]
                    idx = end
                    continue
                pretty_name = graph.nodes[hpo_id].get("name", name)
                matches.append(
                    {
                        "hpo_id": hpo_id,
                        "hpo_name": pretty_name,
                        "matched_span": text[found:end],
                    }
                )
                seen_ids.add(hpo_id)
                masked = masked[:found] + (" " * len(name)) + masked[end:]
                idx = end
            else:
                idx = found + 1
    return matches


def extract_phenotypes_with_llm(
    text: str,
    name_index: dict[str, str],
    graph,
    client: OpenAI,
    *,
    max_candidates: int = 30,
    max_results: int = 12,
) -> list[dict]:
    shortlist = build_phenotype_candidate_shortlist(
        text, name_index, graph, max_candidates=max_candidates
    )
    if not shortlist:
        return extract_phenotypes_from_text(
            text, name_index, graph, max_results=max_results
        )

    payload = {
        "task": (
            "Select phenotype terms that are explicitly supported by the vignette. "
            "Only choose from the provided shortlist."
        ),
        "clinical_vignette": text,
        "candidate_phenotypes": [
            {
                "hpo_id": item["hpo_id"],
                "hpo_name": item["hpo_name"],
                "shared_tokens": item["shared_tokens"],
            }
            for item in shortlist
        ],
        "response_schema": {
            "phenotypes": [
                {
                    "hpo_id": "string",
                    "matched_span": "short phrase from the vignette",
                    "reason": "brief explanation",
                }
            ]
        },
    }

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract phenotype terms from clinical text. "
                    "Only return phenotype IDs from the provided shortlist. "
                    "Return a valid JSON object only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")

    candidate_lookup = {item["hpo_id"]: item["hpo_name"] for item in shortlist}
    extracted: list[dict] = []
    seen_ids: set[str] = set()
    for item in parsed.get("phenotypes", []):
        hpo_id = str(item.get("hpo_id", "")).strip()
        if hpo_id not in candidate_lookup or hpo_id in seen_ids:
            continue
        matched_span = (
            str(item.get("matched_span", "")).strip()
            or candidate_lookup[hpo_id]
        )
        extracted.append(
            {
                "hpo_id": hpo_id,
                "hpo_name": candidate_lookup[hpo_id],
                "matched_span": matched_span,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
        seen_ids.add(hpo_id)
        if len(extracted) >= max_results:
            break

    if extracted:
        return extracted
    return extract_phenotypes_from_text(
        text, name_index, graph, max_results=max_results
    )


def score_candidates_overlap(
    observed_phenotypes: list[dict], profiles: dict[str, dict]
) -> list[dict]:
    observed_ids = {item["hpo_id"] for item in observed_phenotypes}
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
    observed_phenotypes: list[dict],
    profiles: dict[str, dict],
    graph,
    ancestor_distances: dict[str, dict[str, int]],
) -> list[dict]:
    observed_ids = {item["hpo_id"] for item in observed_phenotypes}
    observed_name_lookup = {
        item["hpo_id"]: item["hpo_name"] for item in observed_phenotypes
    }

    # ensure ancestors for any observed IDs not seen at startup
    missing = observed_ids - set(ancestor_distances)
    if missing:
        ancestor_distances = {
            **ancestor_distances,
            **compute_ancestor_distances(graph, missing),
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
            for observed_item in observed_phenotypes:
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
                        "observed_hpo_id": best["observed_hpo_id"],
                        "observed_hpo_name": best["observed_hpo_name"],
                        "disease_hpo_id": disease_hpo_id,
                        "disease_hpo_name": disease_item["hpo_name"],
                        "bridge_ancestor_id": best["ancestor_id"],
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
                "graph_score": round(total_score, 4),
                "normalized_graph_score": round(normalized_graph_score, 4),
                "exact_match_count": len(exact_matches),
                "semantic_match_count": len(semantic_examples),
                "exact_matches": exact_matches[:PROFILE_PHENOTYPES],
                "semantic_examples": semantic_examples[:GRAPH_EXAMPLE_LIMIT],
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


def build_query_text(observed_phenotypes: list[dict]) -> str:
    phenotype_lines = [
        f"- {item['hpo_name']} ({item['hpo_id']})"
        for item in observed_phenotypes
    ]
    return "\n".join(
        [
            "Rare disease differential diagnosis.",
            "Observed phenotypes:",
            *phenotype_lines,
        ]
    )


def retrieve_evidence(
    chroma_collection,
    client: OpenAI,
    observed_phenotypes: list[dict],
    allowed_diseases: set[str],
) -> dict[str, list[dict]]:
    query_text = build_query_text(observed_phenotypes)
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
    mode: Mode,
    graph_scores_by_disease: dict[str, dict] | None,
) -> list[dict]:
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


def build_prompt(
    mode: Mode,
    observed_phenotypes: list[dict],
    candidates: list[dict],
    evidence_by_disease: dict[str, list[dict]],
) -> str:
    task = (
        "You are given a patient's observed phenotype list and the FULL catalogue of "
        "candidate rare diseases. Decide the most likely diagnosis. Some observed "
        "phenotypes may be generalised parent terms or confounders from related "
        "diseases. Return the three most likely diseases, ranked."
    )
    if mode == "rag":
        task = (
            "You are given a patient's observed phenotype list, the FULL candidate "
            "catalogue, and retrieved text evidence per disease. Use the text evidence "
            "to disambiguate. Return the three most likely diseases, ranked."
        )
    elif mode == "rag_graph":
        task = (
            "You are given a patient's observed phenotype list, the FULL candidate "
            "catalogue, retrieved text evidence, and ontology-bridge hints "
            "(`ontology_bridges`) showing how observed phenotypes share an HPO "
            "ancestor with that disease's known phenotypes. Return the three most "
            "likely diseases, ranked."
        )
    payload = {
        "experiment_mode": mode,
        "observed_phenotypes": observed_phenotypes,
        "candidate_diseases": candidates,
        "retrieved_evidence": evidence_by_disease,
        "task": task,
        "response_schema": {
            "ranked_diseases": [
                {"disease_name": "string", "reason": "string"}
            ],
            "final_diagnosis": "string",
            "confidence": "number between 0 and 1",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def rerank_with_llm(
    client: OpenAI,
    mode: Mode,
    observed_phenotypes: list[dict],
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
                    mode, observed_phenotypes, candidates, evidence_by_disease
                ),
            },
        ],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")

    ranked: list[dict] = []
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


@dataclass
class Artefacts:
    profiles: dict[str, dict]
    graph: Any
    ancestor_distances: dict[str, dict[str, int]]
    name_index: dict[str, str]
    chroma_collection: Any
    openai_client: OpenAI


def load_artefacts() -> Artefacts:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot start agent.")
    profiles = load_disease_profiles()
    graph = load_graph()
    ancestor_distances = precompute_ancestor_distances(graph, profiles)
    name_index = build_phenotype_name_index(graph)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    openai_client = OpenAI()
    return Artefacts(
        profiles=profiles,
        graph=graph,
        ancestor_distances=ancestor_distances,
        name_index=name_index,
        chroma_collection=chroma_collection,
        openai_client=openai_client,
    )


def canonicalize_disease_names(
    raw_names: list[str], allowed_names: list[str]
) -> tuple[list[str], list[str]]:
    by_lower = {name.lower(): name for name in allowed_names}
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_names:
        candidate = str(raw_name).strip()
        if not candidate:
            continue
        canonical = by_lower.get(candidate.lower())
        if canonical is None:
            invalid.append(candidate)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        valid.append(canonical)
    return valid, invalid


def build_agentic_patient_payload(
    observed_phenotypes: list[dict], profiles: dict[str, dict]
) -> str:
    catalogue = [
        {
            "disease_name": disease_name,
            "database_id": profile["database_id"],
            "mondo_id": profile["mondo_id"],
        }
        for disease_name, profile in sorted(profiles.items())
    ]
    payload = {
        "task": (
            "Diagnose the most likely disease from this fixed rare-disease catalogue. "
            "Use tools to inspect evidence before finalizing. Use only as many tools "
            "as needed."
        ),
        "observed_phenotypes": observed_phenotypes,
        "candidate_catalogue": catalogue,
        "tool_budget": AGENTIC_TOOL_BUDGET,
        "requirements": [
            "Only choose diseases from the provided catalogue.",
            "Use tool results rather than guessing from disease names alone.",
            "Return exactly 3 ranked diseases in the final structured response.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_agentic_tools(
    observed_phenotypes: list[dict], artefacts: Artefacts
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    allowed_names = sorted(artefacts.profiles)
    all_disease_names = set(allowed_names)
    overlap_scores_cache: list[dict] | None = None
    graph_scores_cache: dict[str, dict] | None = None
    shared_evidence_cache: dict[str, list[dict]] | None = None
    tool_log: list[dict[str, Any]] = []
    retrieval_doc_keys_seen: set[tuple[str, str, str]] = set()
    budget_state = {"budget_hit": False}

    def summarize_profile(disease_name: str) -> dict[str, Any]:
        profile = artefacts.profiles[disease_name]
        return {
            "disease_name": disease_name,
            "database_id": profile["database_id"],
            "mondo_id": profile["mondo_id"],
            "profile_summary": profile["profile_summary"][
                :AGENTIC_PROFILE_TOP_PHENOTYPES
            ],
            "phenotype_count": len(profile["phenotype_ids"]),
        }

    @tool
    def heuristic_overlap(top_k: int = AGENTIC_OVERLAP_TOP_K) -> str:
        """Rank diseases by direct phenotype overlap."""
        nonlocal overlap_scores_cache
        if overlap_scores_cache is None:
            overlap_scores_cache = score_candidates_overlap(
                observed_phenotypes, artefacts.profiles
            )
        rows = [
            {
                "disease_name": item["disease_name"],
                "database_id": item["database_id"],
                "mondo_id": item["mondo_id"],
                "normalized_score": item["normalized_score"],
                "match_score": item["match_score"],
                "exact_matches": item["exact_matches"],
                "matched_phenotypes": [
                    match["hpo_name"] for match in item["exact_match_items"]
                ],
            }
            for item in overlap_scores_cache[: max(1, top_k)]
        ]
        tool_log.append(
            {
                "tool": "heuristic_overlap",
                "top_k": top_k,
                "returned_candidates": len(rows),
            }
        )
        return json.dumps({"top_candidates": rows}, ensure_ascii=False)

    @tool
    def disease_profile_lookup(disease_names: list[str]) -> str:
        """Inspect compact phenotype profiles for named diseases."""
        valid_names, invalid_names = canonicalize_disease_names(
            disease_names, allowed_names
        )
        profiles_payload = [summarize_profile(name) for name in valid_names]
        tool_log.append(
            {
                "tool": "disease_profile_lookup",
                "requested_diseases": disease_names,
                "valid_diseases": valid_names,
                "invalid_diseases": invalid_names,
            }
        )
        return json.dumps(
            {
                "profiles": profiles_payload,
                "invalid_diseases": invalid_names,
            },
            ensure_ascii=False,
        )

    @tool
    def text_evidence_lookup(disease_names: list[str]) -> str:
        """Retrieve text evidence snippets for named diseases."""
        nonlocal shared_evidence_cache
        valid_names, invalid_names = canonicalize_disease_names(
            disease_names, allowed_names
        )
        if shared_evidence_cache is None:
            shared_evidence_cache = retrieve_evidence(
                artefacts.chroma_collection,
                artefacts.openai_client,
                observed_phenotypes,
                all_disease_names,
            )
        evidence_payload = []
        docs_returned = 0
        for disease_name in valid_names:
            snippets = []
            for item in shared_evidence_cache.get(disease_name, []):
                retrieval_doc_keys_seen.add(
                    (
                        disease_name,
                        item.get("source", ""),
                        item.get("title", ""),
                    )
                )
                docs_returned += 1
                snippets.append(
                    {
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "distance": item.get("distance"),
                        "snippet": item.get("document", "")[
                            :AGENTIC_TEXT_EVIDENCE_CHAR_LIMIT
                        ],
                    }
                )
            evidence_payload.append(
                {
                    "disease_name": disease_name,
                    "snippet_count": len(snippets),
                    "snippets": snippets,
                }
            )
        tool_log.append(
            {
                "tool": "text_evidence_lookup",
                "requested_diseases": disease_names,
                "valid_diseases": valid_names,
                "invalid_diseases": invalid_names,
                "docs_returned": docs_returned,
            }
        )
        return json.dumps(
            {
                "evidence": evidence_payload,
                "invalid_diseases": invalid_names,
            },
            ensure_ascii=False,
        )

    @tool
    def relationship_hint_lookup(disease_names: list[str]) -> str:
        """Inspect phenotype relationship hints for named diseases."""
        nonlocal graph_scores_cache
        valid_names, invalid_names = canonicalize_disease_names(
            disease_names, allowed_names
        )
        if graph_scores_cache is None:
            graph_scores = score_candidates_graph(
                observed_phenotypes,
                artefacts.profiles,
                artefacts.graph,
                artefacts.ancestor_distances,
            )
            graph_scores_cache = {
                item["disease_name"]: item for item in graph_scores
            }
        hints_payload = []
        for disease_name in valid_names:
            graph_item = graph_scores_cache.get(disease_name)
            if graph_item is None:
                continue
            hints_payload.append(
                {
                    "disease_name": disease_name,
                    "exact_match_count": graph_item["exact_match_count"],
                    "semantic_match_count": graph_item["semantic_match_count"],
                    "relationship_hints": graph_item["semantic_examples"],
                }
            )
        tool_log.append(
            {
                "tool": "relationship_hint_lookup",
                "requested_diseases": disease_names,
                "valid_diseases": valid_names,
                "invalid_diseases": invalid_names,
                "hinted_diseases": len(hints_payload),
            }
        )
        return json.dumps(
            {
                "relationship_hints": hints_payload,
                "invalid_diseases": invalid_names,
            },
            ensure_ascii=False,
        )

    return (
        [
            heuristic_overlap,
            disease_profile_lookup,
            text_evidence_lookup,
            relationship_hint_lookup,
        ],
        tool_log,
        {
            "budget_state": budget_state,
            "retrieval_doc_keys_seen": retrieval_doc_keys_seen,
        },
    )


def build_agentic_graph(
    tools: list[Any], llm: ChatOpenAI, tool_budget_state: dict[str, Any]
):
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm.with_structured_output(AgenticDiagnosis)
    tool_node = ToolNode(tools)

    def call_model(state: AgenticState) -> dict[str, Any]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def route_after_model(state: AgenticState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(
            last_message, "tool_calls", None
        ):
            tool_message_count = sum(
                1 for message in state["messages"] if message.type == "tool"
            )
            if tool_message_count >= AGENTIC_TOOL_BUDGET:
                tool_budget_state["budget_hit"] = True
                return "finalize"
            return "tools"
        return "finalize"

    def finalize(state: AgenticState) -> dict[str, Any]:
        formatter_prompt = SystemMessage(
            content=(
                "Use the full conversation and tool outputs to produce the final ranked "
                "diagnosis. Only select diseases from the provided catalogue. Return "
                "exactly 3 ranked diseases."
            )
        )
        structured = structured_llm.invoke(
            [formatter_prompt, *state["messages"]]
        )
        structured_payload = (
            structured.model_dump()
            if hasattr(structured, "model_dump")
            else dict(structured)
        )
        return {"structured_response": structured_payload}

    graph = StateGraph(AgenticState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_model,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def normalize_agentic_response(
    structured_response: dict[str, Any],
    allowed_names: list[str],
    heuristic_ranking_full: list[str],
) -> dict[str, Any]:
    valid_ranked: list[dict[str, str]] = []
    seen: set[str] = set()
    valid_names, _ = canonicalize_disease_names(
        [
            item.get("disease_name", "")
            for item in structured_response.get("ranked_diseases", [])
        ],
        allowed_names,
    )
    reason_lookup = {
        str(item.get("disease_name", ""))
        .strip()
        .lower(): str(item.get("reason", ""))
        .strip()
        for item in structured_response.get("ranked_diseases", [])
    }
    for disease_name in valid_names:
        if disease_name in seen:
            continue
        seen.add(disease_name)
        valid_ranked.append(
            {
                "disease_name": disease_name,
                "reason": reason_lookup.get(disease_name.lower(), ""),
            }
        )
        if len(valid_ranked) >= 3:
            break

    for disease_name in heuristic_ranking_full:
        if len(valid_ranked) >= 3:
            break
        if disease_name in seen:
            continue
        seen.add(disease_name)
        valid_ranked.append(
            {
                "disease_name": disease_name,
                "reason": "Backfilled from heuristic overlap ranking.",
            }
        )

    final_diagnosis_raw = str(
        structured_response.get("final_diagnosis", "")
    ).strip()
    canonical_final, _ = canonicalize_disease_names(
        [final_diagnosis_raw], allowed_names
    )
    final_diagnosis = (
        canonical_final[0]
        if canonical_final
        else valid_ranked[0]["disease_name"]
    )
    try:
        confidence = float(structured_response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "final_diagnosis": final_diagnosis,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "ranked_diseases": valid_ranked[:3],
    }


def diagnose_agentic(
    observed_phenotypes: list[dict], artefacts: Artefacts, *, top_k: int = 5
) -> dict:
    trace = Trace()
    overlap_scores = score_candidates_overlap(
        observed_phenotypes, artefacts.profiles
    )
    heuristic_ranking_full = [item["disease_name"] for item in overlap_scores]

    tools, tool_log, tool_state = build_agentic_tools(
        observed_phenotypes, artefacts
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    graph = build_agentic_graph(tools, llm, tool_state["budget_state"])
    initial_messages = [
        SystemMessage(
            content=(
                "You are an autonomous rare-disease diagnosis agent. "
                "Use tools to inspect evidence before finalizing. "
                "Choose only from the provided disease catalogue. "
                "Avoid unnecessary tool calls."
            )
        ),
        HumanMessage(
            content=build_agentic_patient_payload(
                observed_phenotypes, artefacts.profiles
            )
        ),
    ]

    with trace.step(
        "agentic_loop",
        "Agentic tool-selection loop",
        f"langgraph:{CHAT_MODEL}",
        {
            "observed_phenotype_count": len(observed_phenotypes),
            "tool_budget": AGENTIC_TOOL_BUDGET,
        },
    ) as event:
        result = graph.invoke(
            {"messages": initial_messages},
            config={"recursion_limit": AGENTIC_RECURSION_LIMIT},
        )
        structured_response = result.get("structured_response", {})
        normalized_response = normalize_agentic_response(
            structured_response,
            sorted(artefacts.profiles),
            heuristic_ranking_full,
        )
        event.output_summary = {
            "predicted_diagnosis": normalized_response["final_diagnosis"],
            "confidence": normalized_response["confidence"],
            "total_tool_calls": len(tool_log),
            "tools_used": [item["tool"] for item in tool_log],
            "tool_budget_hit": bool(tool_state["budget_state"]["budget_hit"]),
        }

    for index, item in enumerate(tool_log, start=1):
        trace.events.append(
            TraceEvent(
                step=f"tool_{index}",
                label=f"Tool call {index}: {item['tool']}",
                tool=item["tool"],
                started_at=0.0,
                duration_ms=0.0,
                input_summary={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "tool",
                        "returned_candidates",
                        "docs_returned",
                        "hinted_diseases",
                        "valid_diseases",
                    }
                },
                output_summary={
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "returned_candidates",
                        "docs_returned",
                        "hinted_diseases",
                        "valid_diseases",
                    }
                },
            )
        )

    overlap_top = [item["disease_name"] for item in overlap_scores[:top_k]]
    heuristic_match = (
        overlap_top[0] == normalized_response["final_diagnosis"]
        if overlap_top
        else False
    )

    graph_scores_by_disease = None
    if any(item["tool"] == "relationship_hint_lookup" for item in tool_log):
        graph_scores = score_candidates_graph(
            observed_phenotypes,
            artefacts.profiles,
            artefacts.graph,
            artefacts.ancestor_distances,
        )
        graph_scores_by_disease = {
            item["disease_name"]: item for item in graph_scores
        }

    evidence_by_disease = {}
    if any(item["tool"] == "text_evidence_lookup" for item in tool_log):
        evidence_by_disease = retrieve_evidence(
            artefacts.chroma_collection,
            artefacts.openai_client,
            observed_phenotypes,
            set(artefacts.profiles),
        )

    enriched_ranked = []
    for entry in normalized_response["ranked_diseases"]:
        disease_name = entry["disease_name"]
        overlap_match = next(
            (
                item
                for item in overlap_scores
                if item["disease_name"] == disease_name
            ),
            None,
        )
        graph_match = (
            graph_scores_by_disease.get(disease_name)
            if graph_scores_by_disease
            else None
        )
        enriched_ranked.append(
            {
                "disease_name": disease_name,
                "reason": entry["reason"],
                "exact_phenotype_matches": (
                    overlap_match["exact_match_items"] if overlap_match else []
                ),
                "match_score": (
                    overlap_match["match_score"] if overlap_match else 0.0
                ),
                "evidence": evidence_by_disease.get(disease_name, []),
                "ontology_bridges": (
                    graph_match["semantic_examples"] if graph_match else []
                ),
                "profile_summary": artefacts.profiles[disease_name][
                    "profile_summary"
                ],
                "database_id": artefacts.profiles[disease_name]["database_id"],
                "mondo_id": artefacts.profiles[disease_name]["mondo_id"],
            }
        )

    return {
        "mode": "agentic",
        "observed_phenotypes": observed_phenotypes,
        "final_diagnosis": normalized_response["final_diagnosis"],
        "confidence": normalized_response["confidence"],
        "ranked_diseases": enriched_ranked,
        "heuristic_top_k": overlap_top,
        "heuristic_agrees_with_llm": heuristic_match,
        "graph_scores_by_disease": graph_scores_by_disease,
        "evidence_by_disease": evidence_by_disease,
        "trace": trace.to_list(),
    }


def diagnose(
    observed_phenotypes: list[dict],
    mode: Mode,
    artefacts: Artefacts,
    *,
    top_k: int = 5,
) -> dict:
    trace = Trace()

    if not observed_phenotypes:
        raise ValueError("Need at least one observed phenotype.")

    if mode == "agentic":
        return diagnose_agentic(
            observed_phenotypes,
            artefacts,
            top_k=top_k,
        )

    with trace.step(
        "score_overlap",
        "Heuristic phenotype overlap (always run for comparison)",
        "score_candidates_overlap",
        {"observed_phenotype_count": len(observed_phenotypes)},
    ) as event:
        overlap_scores = score_candidates_overlap(
            observed_phenotypes, artefacts.profiles
        )
        event.output_summary = {
            "top_disease": (
                overlap_scores[0]["disease_name"] if overlap_scores else None
            ),
            "top_normalized_score": (
                overlap_scores[0]["normalized_score"]
                if overlap_scores
                else 0.0
            ),
            "top_exact_matches": (
                overlap_scores[0]["exact_matches"] if overlap_scores else 0
            ),
        }

    graph_scores_by_disease: dict[str, dict] | None = None
    if mode == "rag_graph":
        with trace.step(
            "score_graph",
            "Ontology bridge scoring over HPO `is_a` graph",
            "score_candidates_graph",
            {"observed_phenotype_count": len(observed_phenotypes)},
        ) as event:
            graph_scores = score_candidates_graph(
                observed_phenotypes,
                artefacts.profiles,
                artefacts.graph,
                artefacts.ancestor_distances,
            )
            graph_scores_by_disease = {
                item["disease_name"]: item for item in graph_scores
            }
            event.output_summary = {
                "top_disease": (
                    graph_scores[0]["disease_name"] if graph_scores else None
                ),
                "top_normalized_graph_score": (
                    graph_scores[0]["normalized_graph_score"]
                    if graph_scores
                    else 0.0
                ),
                "top_semantic_match_count": (
                    graph_scores[0]["semantic_match_count"]
                    if graph_scores
                    else 0
                ),
            }

    evidence_by_disease: dict[str, list[dict]] = {}
    if mode in {"rag", "rag_graph"}:
        with trace.step(
            "retrieve_evidence",
            "Embed patient profile and retrieve from MedQuAD + PubMed (Chroma)",
            "retrieve_evidence",
            {"k": RETRIEVAL_K, "evidence_per_disease": EVIDENCE_PER_DISEASE},
        ) as event:
            evidence_by_disease = retrieve_evidence(
                artefacts.chroma_collection,
                artefacts.openai_client,
                observed_phenotypes,
                set(artefacts.profiles),
            )
            event.output_summary = {
                "diseases_with_evidence": len(evidence_by_disease),
                "total_snippets": sum(
                    len(v) for v in evidence_by_disease.values()
                ),
            }

    with trace.step(
        "build_candidates",
        "Assemble alphabetised candidate catalogue",
        "build_full_candidate_list",
        {"mode": mode, "num_diseases": len(artefacts.profiles)},
    ) as event:
        candidates = build_full_candidate_list(
            artefacts.profiles, mode, graph_scores_by_disease
        )
        event.output_summary = {
            "candidates": len(candidates),
            "with_ontology_bridges": sum(
                1 for c in candidates if c.get("ontology_bridges")
            ),
        }

    with trace.step(
        "llm_rerank",
        f"LLM ranks the catalogue ({CHAT_MODEL}, temperature=0)",
        f"openai.chat.completions:{CHAT_MODEL}",
        {"mode": mode, "num_candidates": len(candidates)},
    ) as event:
        llm_result = rerank_with_llm(
            artefacts.openai_client,
            mode,
            observed_phenotypes,
            candidates,
            evidence_by_disease,
        )
        event.output_summary = {
            "predicted_diagnosis": llm_result["final_diagnosis"],
            "confidence": llm_result["confidence"],
            "ranked_diseases": [
                d["disease_name"] for d in llm_result["ranked_diseases"]
            ],
        }

    overlap_top = [item["disease_name"] for item in overlap_scores[:top_k]]
    heuristic_match = (
        overlap_top[0] == llm_result["final_diagnosis"]
        if overlap_top
        else False
    )

    enriched_ranked = []
    for entry in llm_result["ranked_diseases"]:
        disease_name = entry["disease_name"]
        overlap_match = next(
            (
                item
                for item in overlap_scores
                if item["disease_name"] == disease_name
            ),
            None,
        )
        graph_match = (
            graph_scores_by_disease.get(disease_name)
            if graph_scores_by_disease
            else None
        )
        enriched_ranked.append(
            {
                "disease_name": disease_name,
                "reason": entry["reason"],
                "exact_phenotype_matches": (
                    overlap_match["exact_match_items"] if overlap_match else []
                ),
                "match_score": (
                    overlap_match["match_score"] if overlap_match else 0.0
                ),
                "evidence": evidence_by_disease.get(disease_name, []),
                "ontology_bridges": (
                    graph_match["semantic_examples"] if graph_match else []
                ),
                "profile_summary": artefacts.profiles[disease_name][
                    "profile_summary"
                ],
                "database_id": artefacts.profiles[disease_name]["database_id"],
                "mondo_id": artefacts.profiles[disease_name]["mondo_id"],
            }
        )

    return {
        "mode": mode,
        "observed_phenotypes": observed_phenotypes,
        "final_diagnosis": llm_result["final_diagnosis"],
        "confidence": llm_result["confidence"],
        "ranked_diseases": enriched_ranked,
        "heuristic_top_k": overlap_top,
        "heuristic_agrees_with_llm": heuristic_match,
        "graph_scores_by_disease": graph_scores_by_disease,
        "evidence_by_disease": evidence_by_disease,
        "trace": trace.to_list(),
    }
