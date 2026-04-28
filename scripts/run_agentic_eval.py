import json
import os
import sys
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import dotenv_values
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.diagnose import (  # noqa: E402
    CHAT_MODEL,
    load_artefacts,
    load_patients,
    retrieve_evidence,
    score_candidates_graph,
    score_candidates_overlap,
)


ENV_FILE = ROOT / ".env"
ENV_VALUES = dotenv_values(ENV_FILE)
if ENV_VALUES.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ENV_VALUES["OPENAI_API_KEY"]

OUTPUT_DIR = ROOT / "outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "agentic_eval_predictions.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "agentic_eval_summary.csv"
METRICS_PATH = OUTPUT_DIR / "agentic_eval_metrics.json"
FIXED_SUITE_METRICS_PATH = OUTPUT_DIR / "agent_eval_suite_metrics.json"

MAX_PATIENTS = int(
    os.getenv(
        "AGENTIC_EVAL_MAX_PATIENTS", os.getenv("AGENT_EVAL_MAX_PATIENTS", "0")
    )
)
TOOL_BUDGET = int(os.getenv("AGENTIC_TOOL_BUDGET", "6"))
RECURSION_LIMIT = int(
    os.getenv("AGENTIC_RECURSION_LIMIT", str((TOOL_BUDGET * 2) + 4))
)
OVERLAP_TOOL_TOP_K = int(os.getenv("AGENTIC_OVERLAP_TOP_K", "8"))
PROFILE_TOOL_TOP_PHENOTYPES = int(
    os.getenv("AGENTIC_PROFILE_TOP_PHENOTYPES", "8")
)
TEXT_EVIDENCE_CHAR_LIMIT = int(
    os.getenv("AGENTIC_TEXT_EVIDENCE_CHAR_LIMIT", "500")
)


class RankedDisease(BaseModel):
    disease_name: str
    reason: str


class AgenticDiagnosis(BaseModel):
    final_diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)
    ranked_diseases: list[RankedDisease] = Field(min_length=3, max_length=3)


class AgenticState(MessagesState):
    structured_response: dict[str, Any]


def load_eval_patients() -> list[dict]:
    patients = load_patients()
    if MAX_PATIENTS > 0:
        return patients[:MAX_PATIENTS]
    return patients


def reciprocal_rank(ranking: list[str], target: str) -> float:
    for index, name in enumerate(ranking, start=1):
        if name == target:
            return 1.0 / index
    return 0.0


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


def build_patient_payload(patient: dict, profiles: dict[str, dict]) -> str:
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
            "You must inspect evidence with tools before finalizing. Use only as many "
            "tools as needed. After each tool result, decide whether more evidence is "
            "actually necessary."
        ),
        "patient_id": patient["patient_id"],
        "observed_phenotypes": patient["all_phenotypes"],
        "candidate_catalogue": catalogue,
        "tool_budget": TOOL_BUDGET,
        "requirements": [
            "Only choose diseases from the provided catalogue.",
            "Use tool results rather than guessing from names alone.",
            "Return exactly 3 ranked diseases in the final structured response.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_patient_tools(
    patient: dict, artefacts
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
                :PROFILE_TOOL_TOP_PHENOTYPES
            ],
            "phenotype_count": len(profile["phenotype_ids"]),
        }

    @tool
    def heuristic_overlap(top_k: int = OVERLAP_TOOL_TOP_K) -> str:
        """Rank diseases by direct phenotype overlap. Use this to identify strong
        candidates before deciding whether more evidence is necessary."""
        nonlocal overlap_scores_cache
        if overlap_scores_cache is None:
            overlap_scores_cache = score_candidates_overlap(
                patient["all_phenotypes"], artefacts.profiles
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
        """Inspect compact phenotype profiles for specific diseases."""
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
        """Retrieve text evidence snippets for named diseases using the patient's
        phenotype query against the indexed corpus."""
        nonlocal shared_evidence_cache
        valid_names, invalid_names = canonicalize_disease_names(
            disease_names, allowed_names
        )
        if shared_evidence_cache is None:
            shared_evidence_cache = retrieve_evidence(
                artefacts.chroma_collection,
                artefacts.openai_client,
                patient["all_phenotypes"],
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
                            :TEXT_EVIDENCE_CHAR_LIMIT
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
        """Inspect phenotype relationship hints for named diseases based on shared
        parent concepts in the phenotype graph."""
        nonlocal graph_scores_cache
        valid_names, invalid_names = canonicalize_disease_names(
            disease_names, allowed_names
        )
        if graph_scores_cache is None:
            graph_scores = score_candidates_graph(
                patient["all_phenotypes"],
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


def build_agent_graph(
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
            if tool_message_count >= TOOL_BUDGET:
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
        if hasattr(structured, "model_dump"):
            structured_payload = structured.model_dump()
        else:
            structured_payload = dict(structured)
        return {"structured_response": structured_payload}

    graph = StateGraph(AgenticState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_model,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
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
    if canonical_final:
        final_diagnosis = canonical_final[0]
    else:
        final_diagnosis = valid_ranked[0]["disease_name"]

    try:
        confidence = float(structured_response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "final_diagnosis": final_diagnosis,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "ranked_diseases": valid_ranked[:3],
    }


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_patients = len(records)
    if num_patients == 0:
        return {"num_patients": 0}

    heuristic_top1 = sum(
        1
        for row in records
        if row["heuristic_ranked_diseases"][:1] == [row["true_diagnosis"]]
    )
    heuristic_top3 = sum(
        1
        for row in records
        if row["true_diagnosis"] in row["heuristic_ranked_diseases"][:3]
    )
    heuristic_mrr = (
        sum(
            reciprocal_rank(
                row["heuristic_ranked_diseases"], row["true_diagnosis"]
            )
            for row in records
        )
        / num_patients
    )

    agentic_top1 = sum(
        1
        for row in records
        if row["agentic_ranked_diseases"][:1] == [row["true_diagnosis"]]
    )
    agentic_top3 = sum(
        1
        for row in records
        if row["true_diagnosis"] in row["agentic_ranked_diseases"][:3]
    )
    agentic_mrr = (
        sum(
            reciprocal_rank(
                row["agentic_ranked_diseases"], row["true_diagnosis"]
            )
            for row in records
        )
        / num_patients
    )

    return {
        "num_patients": num_patients,
        "heuristic_top1_accuracy": round(heuristic_top1 / num_patients, 4),
        "heuristic_top3_accuracy": round(heuristic_top3 / num_patients, 4),
        "heuristic_mrr": round(heuristic_mrr, 4),
        "agentic_top1_accuracy": round(agentic_top1 / num_patients, 4),
        "agentic_top3_accuracy": round(agentic_top3 / num_patients, 4),
        "agentic_mrr": round(agentic_mrr, 4),
        "delta_top1_agentic_minus_heuristic": round(
            (agentic_top1 - heuristic_top1) / num_patients, 4
        ),
        "delta_top3_agentic_minus_heuristic": round(
            (agentic_top3 - heuristic_top3) / num_patients, 4
        ),
        "delta_mrr_agentic_minus_heuristic": round(
            agentic_mrr - heuristic_mrr, 4
        ),
        "mean_confidence": round(
            sum(row["confidence"] for row in records) / num_patients, 4
        ),
        "mean_tool_calls": round(
            sum(row["total_tool_calls"] for row in records) / num_patients, 3
        ),
        "mean_unique_tools_used": round(
            sum(row["unique_tools_used"] for row in records) / num_patients, 3
        ),
        "mean_unique_retrieved_docs": round(
            sum(row["unique_retrieved_docs"] for row in records)
            / num_patients,
            3,
        ),
        "fraction_using_overlap_tool": round(
            sum(1 for row in records if row["used_overlap_tool"])
            / num_patients,
            4,
        ),
        "fraction_using_profile_tool": round(
            sum(1 for row in records if row["used_profile_tool"])
            / num_patients,
            4,
        ),
        "fraction_using_text_tool": round(
            sum(1 for row in records if row["used_text_tool"]) / num_patients,
            4,
        ),
        "fraction_using_relationship_tool": round(
            sum(1 for row in records if row["used_relationship_tool"])
            / num_patients,
            4,
        ),
        "fraction_tool_budget_hit": round(
            sum(1 for row in records if row["tool_budget_hit"]) / num_patients,
            4,
        ),
    }


def maybe_load_fixed_suite_metrics() -> dict[str, Any] | None:
    if not FIXED_SUITE_METRICS_PATH.exists():
        return None
    data = json.loads(FIXED_SUITE_METRICS_PATH.read_text(encoding="utf-8"))
    metrics_by_mode = data.get("metrics_by_mode", {})
    return {
        mode: {
            "top1_accuracy": metrics_by_mode.get(mode, {}).get(
                "llm_top1_accuracy"
            ),
            "top3_accuracy": metrics_by_mode.get(mode, {}).get(
                "llm_top3_accuracy"
            ),
            "mrr": metrics_by_mode.get(mode, {}).get("llm_mrr"),
        }
        for mode in ("static", "rag", "rag_graph")
        if mode in metrics_by_mode
    }


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    patients = load_eval_patients()
    artefacts = load_artefacts()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    records: list[dict[str, Any]] = []
    allowed_names = sorted(artefacts.profiles)

    for patient_index, patient in enumerate(patients, start=1):
        overlap_scores = score_candidates_overlap(
            patient["all_phenotypes"], artefacts.profiles
        )
        heuristic_ranking_full = [
            item["disease_name"] for item in overlap_scores
        ]

        tools, tool_log, tool_state = build_patient_tools(patient, artefacts)
        graph = build_agent_graph(tools, llm, tool_state["budget_state"])
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
                content=build_patient_payload(patient, artefacts.profiles)
            ),
        ]

        error_message = None
        final_message_content = ""
        normalized_response = None
        try:
            result = graph.invoke(
                {"messages": initial_messages},
                config={"recursion_limit": RECURSION_LIMIT},
            )
            structured_response = result.get("structured_response", {})
            normalized_response = normalize_agentic_response(
                structured_response,
                allowed_names,
                heuristic_ranking_full,
            )
            messages = result.get("messages", [])
            for message in reversed(messages):
                if isinstance(message, AIMessage) and message.content:
                    final_message_content = str(message.content)
                    break
        except Exception as exc:
            error_message = str(exc)
            normalized_response = {
                "final_diagnosis": heuristic_ranking_full[0],
                "confidence": 0.0,
                "ranked_diseases": [
                    {
                        "disease_name": disease_name,
                        "reason": "Fallback to heuristic ranking after agent failure.",
                    }
                    for disease_name in heuristic_ranking_full[:3]
                ],
            }

        tool_names = [item["tool"] for item in tool_log]
        row = {
            "patient_id": patient["patient_id"],
            "true_diagnosis": patient["true_diagnosis"],
            "database_id": patient["database_id"],
            "mondo_id": patient["mondo_id"],
            "heuristic_ranked_diseases": heuristic_ranking_full,
            "agentic_ranked_diseases": [
                item["disease_name"]
                for item in normalized_response["ranked_diseases"]
            ],
            "predicted_diagnosis": normalized_response["final_diagnosis"],
            "confidence": normalized_response["confidence"],
            "tool_log": tool_log,
            "tool_sequence": tool_names,
            "total_tool_calls": len(tool_names),
            "unique_tools_used": len(set(tool_names)),
            "used_overlap_tool": "heuristic_overlap" in tool_names,
            "used_profile_tool": "disease_profile_lookup" in tool_names,
            "used_text_tool": "text_evidence_lookup" in tool_names,
            "used_relationship_tool": "relationship_hint_lookup" in tool_names,
            "unique_retrieved_docs": len(
                tool_state["retrieval_doc_keys_seen"]
            ),
            "tool_budget_hit": bool(tool_state["budget_state"]["budget_hit"]),
            "final_message_content": final_message_content,
            "error": error_message,
        }
        records.append(row)
        print(
            f"[{patient_index}/{len(patients)}][agentic] {row['patient_id']}: "
            f"true={row['true_diagnosis']} pred={row['predicted_diagnosis']} "
            f"tools={row['total_tool_calls']}"
        )

    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = [
        {
            "patient_id": row["patient_id"],
            "true_diagnosis": row["true_diagnosis"],
            "predicted_diagnosis": row["predicted_diagnosis"],
            "heuristic_top1_correct": row["heuristic_ranked_diseases"][:1]
            == [row["true_diagnosis"]],
            "heuristic_top3_correct": row["true_diagnosis"]
            in row["heuristic_ranked_diseases"][:3],
            "agentic_top1_correct": row["agentic_ranked_diseases"][:1]
            == [row["true_diagnosis"]],
            "agentic_top3_correct": row["true_diagnosis"]
            in row["agentic_ranked_diseases"][:3],
            "confidence": row["confidence"],
            "total_tool_calls": row["total_tool_calls"],
            "used_overlap_tool": row["used_overlap_tool"],
            "used_profile_tool": row["used_profile_tool"],
            "used_text_tool": row["used_text_tool"],
            "used_relationship_tool": row["used_relationship_tool"],
            "unique_retrieved_docs": row["unique_retrieved_docs"],
            "tool_budget_hit": row["tool_budget_hit"],
            "error": row["error"] or "",
        }
        for row in records
    ]
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    metrics = {
        "config": {
            "mode": "agentic_tool_selection",
            "num_patients": len(records),
            "tool_budget": TOOL_BUDGET,
            "recursion_limit": RECURSION_LIMIT,
            "chat_model": CHAT_MODEL,
        },
        "metrics": compute_metrics(records),
    }
    fixed_suite_metrics = maybe_load_fixed_suite_metrics()
    if fixed_suite_metrics is not None:
        metrics["fixed_suite_reference"] = fixed_suite_metrics

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
