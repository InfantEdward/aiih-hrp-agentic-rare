import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
ENV_VALUES = dotenv_values(ENV_FILE)
for key, value in ENV_VALUES.items():
    if value is not None and key not in os.environ:
        os.environ[key] = value

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATIONS_PATH = OUTPUT_DIR / "locked_hpoa_annotations.csv"
GRAPH_PATH = OUTPUT_DIR / "locked_disease_graph.pkl"
SIMULATED_JSONL = OUTPUT_DIR / "simulated_patients.jsonl"
SIMULATED_SUMMARY = OUTPUT_DIR / "simulated_patients_summary.csv"

SEED = int(os.getenv("SIM_SEED", "42"))
PATIENTS_PER_DISEASE = int(os.getenv("PATIENTS_PER_DISEASE", "8"))
MIN_TRUE_PHENOTYPES = int(os.getenv("SIM_MIN_TRUE_PHENOTYPES", "3"))
MAX_RANDOM_NOISE_PHENOTYPES = int(
    os.getenv("SIM_MAX_RANDOM_NOISE_PHENOTYPES", "4")
)
MAX_CORRUPTIONS = int(os.getenv("SIM_MAX_CORRUPTIONS", "2"))
GENERAL_DROPOUT_PROB = float(os.getenv("SIM_GENERAL_DROPOUT_PROB", "0.4"))
SIGNATURE_TOP_K = int(os.getenv("SIM_SIGNATURE_TOP_K", "3"))
SIGNATURE_DROPOUT_PROB = float(os.getenv("SIM_SIGNATURE_DROPOUT_PROB", "0.85"))
MIN_OBSERVED_PHENOTYPES = int(os.getenv("SIM_MIN_OBSERVED_PHENOTYPES", "4"))

MAX_PEER_CONFOUNDERS = int(os.getenv("SIM_MAX_PEER_CONFOUNDERS", "6"))
MIN_PEER_CONFOUNDERS = int(os.getenv("SIM_MIN_PEER_CONFOUNDERS", "3"))
GENERALIZATION_PROB = float(os.getenv("SIM_GENERALIZATION_PROB", "0.5"))
MAX_GENERALIZATION_HOPS = int(os.getenv("SIM_MAX_GENERALIZATION_HOPS", "2"))
GENERALIZATION_MIN_DEPTH = int(os.getenv("SIM_GENERALIZATION_MIN_DEPTH", "3"))


def load_graph():
    if not GRAPH_PATH.exists():
        return None
    with GRAPH_PATH.open("rb") as handle:
        return pickle.load(handle)


def build_sibling_map(graph) -> dict[str, list[tuple[str, str]]]:
    sibling_map: dict[str, set[tuple[str, str]]] = defaultdict(set)
    if graph is None:
        return {}

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "phenotype":
            continue
        parents = [
            parent
            for _, parent, edge_data in graph.out_edges(node_id, data=True)
            if edge_data.get("relation") == "is_a"
        ]
        for parent_id in parents:
            siblings = [
                child
                for child, _, edge_data in graph.in_edges(parent_id, data=True)
                if edge_data.get("relation") == "is_a" and child != node_id
            ]
            for sibling_id in siblings:
                sibling_name = graph.nodes[sibling_id].get("name", sibling_id)
                sibling_map[node_id].add((sibling_id, sibling_name))

    return {key: sorted(value) for key, value in sibling_map.items()}


def build_peer_confounder_pool(
    annotations: pd.DataFrame,
) -> dict[str, list[tuple[str, str, str]]]:
    by_disease: dict[str, list[tuple[str, str]]] = {}
    for disease_name, rows in annotations.groupby("disease_name"):
        deduped = rows.drop_duplicates("hpo_id")
        by_disease[disease_name] = [
            (row.hpo_id, row.hpo_name)
            for row in deduped.itertuples(index=False)
        ]

    peer_pool: dict[str, list[tuple[str, str, str]]] = {}
    for disease_name in by_disease:
        own_ids = {hpo_id for hpo_id, _ in by_disease[disease_name]}
        peers: list[tuple[str, str, str]] = []
        for other_disease, items in by_disease.items():
            if other_disease == disease_name:
                continue
            for hpo_id, hpo_name in items:
                if hpo_id in own_ids:
                    continue
                peers.append((hpo_id, hpo_name, other_disease))
        peer_pool[disease_name] = peers
    return peer_pool


def sample_peer_confounders(
    rng: np.random.Generator,
    disease_name: str,
    peer_pool: dict[str, list[tuple[str, str, str]]],
    excluded_ids: set[str],
) -> list[dict]:
    options = [
        item
        for item in peer_pool.get(disease_name, [])
        if item[0] not in excluded_ids
    ]
    if not options or MAX_PEER_CONFOUNDERS <= 0:
        return []

    low = min(MIN_PEER_CONFOUNDERS, MAX_PEER_CONFOUNDERS)
    high = MAX_PEER_CONFOUNDERS
    count = int(rng.integers(low, high + 1))
    count = min(count, len(options))
    if count <= 0:
        return []

    indices = rng.choice(len(options), size=count, replace=False)
    return [
        {
            "hpo_id": options[int(idx)][0],
            "hpo_name": options[int(idx)][1],
            "source_disease": options[int(idx)][2],
        }
        for idx in indices
    ]


def get_ancestor_chain(
    graph, hpo_id: str, max_hops: int
) -> list[tuple[str, str]]:
    if graph is None or hpo_id not in graph:
        return []
    chain: list[tuple[str, str]] = []
    current = hpo_id
    for _ in range(max_hops):
        parents = [
            parent
            for _, parent, edge_data in graph.out_edges(current, data=True)
            if edge_data.get("relation") == "is_a"
        ]
        if not parents:
            break
        current = parents[0]
        chain.append((current, graph.nodes[current].get("name", current)))
    return chain


def generalize_phenotypes(
    rng: np.random.Generator,
    observed_rows: list[dict],
    graph,
    used_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    if graph is None or GENERALIZATION_PROB <= 0:
        return observed_rows, []

    new_rows: list[dict] = []
    log: list[dict] = []
    for row in observed_rows:
        hpo_id = row["hpo_id"]
        if rng.random() >= GENERALIZATION_PROB:
            new_rows.append(row)
            continue
        chain = get_ancestor_chain(graph, hpo_id, MAX_GENERALIZATION_HOPS)
        # Filter out ancestors that are too generic — require the ancestor itself
        # has at least GENERALIZATION_MIN_DEPTH ancestors above it.
        viable = []
        for ancestor_id, ancestor_name in chain:
            depth_above = len(
                get_ancestor_chain(
                    graph, ancestor_id, GENERALIZATION_MIN_DEPTH + 1
                )
            )
            if depth_above >= GENERALIZATION_MIN_DEPTH:
                viable.append((ancestor_id, ancestor_name))
        if not viable:
            new_rows.append(row)
            continue
        ancestor_id, ancestor_name = viable[int(rng.integers(0, len(viable)))]
        if ancestor_id in used_ids:
            new_rows.append(row)
            continue
        used_ids.add(ancestor_id)
        new_rows.append({"hpo_id": ancestor_id, "hpo_name": ancestor_name})
        log.append(
            {
                "source_hpo_id": hpo_id,
                "source_hpo_name": row["hpo_name"],
                "generalized_hpo_id": ancestor_id,
                "generalized_hpo_name": ancestor_name,
            }
        )
    return new_rows, log


def ensure_minimum_selection(
    selected: pd.DataFrame, phenotype_rows: pd.DataFrame, minimum: int
) -> pd.DataFrame:
    if len(selected) >= minimum:
        return selected
    top_rows = phenotype_rows.sort_values(
        "frequency_probability", ascending=False
    ).head(minimum)
    combined = pd.concat([selected, top_rows], ignore_index=True)
    return combined.drop_duplicates("hpo_id")


def apply_dropout(
    rng: np.random.Generator, selected: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    ranked = selected.sort_values(
        ["frequency_probability", "hpo_id"], ascending=[False, True]
    ).reset_index(drop=True)
    keep_mask = []
    dropped_rows = []

    for index, row in ranked.iterrows():
        dropout_prob = GENERAL_DROPOUT_PROB
        if index < SIGNATURE_TOP_K:
            dropout_prob = max(dropout_prob, SIGNATURE_DROPOUT_PROB)
        keep = rng.random() >= dropout_prob
        keep_mask.append(keep)
        if not keep:
            dropped_rows.append(
                {
                    "hpo_id": row.hpo_id,
                    "hpo_name": row.hpo_name,
                    "frequency_probability": round(
                        float(row.frequency_probability), 3
                    ),
                }
            )

    kept = ranked[np.array(keep_mask, dtype=bool)].copy()
    kept = ensure_minimum_selection(kept, ranked, MIN_OBSERVED_PHENOTYPES)
    kept = kept.drop_duplicates("hpo_id")
    dropped_hpo_ids = {item["hpo_id"] for item in dropped_rows}
    dropped_rows = [
        item
        for item in dropped_rows
        if item["hpo_id"] not in set(kept["hpo_id"])
    ]
    return kept, dropped_rows


def sample_corruptions(
    rng: np.random.Generator,
    observed_rows: pd.DataFrame,
    disease_hpo_ids: set[str],
    sibling_map: dict[str, list[tuple[str, str]]],
) -> tuple[list[dict], set[str]]:
    if MAX_CORRUPTIONS <= 0 or observed_rows.empty:
        return [], set()

    candidates = observed_rows.to_dict(orient="records")
    rng.shuffle(candidates)
    corruptions = []
    replaced_ids: set[str] = set()
    used_corrupted_hpo_ids: set[str] = set()

    for row in candidates:
        if len(corruptions) >= MAX_CORRUPTIONS:
            break
        sibling_options = [
            sibling
            for sibling in sibling_map.get(row["hpo_id"], [])
            if sibling[0] not in disease_hpo_ids
            and sibling[0] not in used_corrupted_hpo_ids
        ]
        if not sibling_options:
            continue
        sibling_id, sibling_name = sibling_options[
            int(rng.integers(0, len(sibling_options)))
        ]
        corruptions.append(
            {
                "source_hpo_id": row["hpo_id"],
                "source_hpo_name": row["hpo_name"],
                "hpo_id": sibling_id,
                "hpo_name": sibling_name,
            }
        )
        replaced_ids.add(row["hpo_id"])
        used_corrupted_hpo_ids.add(sibling_id)

    return corruptions, replaced_ids


def sample_random_noise(
    rng: np.random.Generator,
    disease_hpo_ids: set[str],
    excluded_ids: set[str],
    noise_pool: list[tuple[str, str]],
) -> list[dict]:
    noise_count = int(rng.integers(0, MAX_RANDOM_NOISE_PHENOTYPES + 1))
    if noise_count <= 0:
        return []

    valid_noise = [
        item
        for item in noise_pool
        if item[0] not in disease_hpo_ids and item[0] not in excluded_ids
    ]
    if not valid_noise:
        return []

    picked = rng.choice(
        valid_noise, size=min(noise_count, len(valid_noise)), replace=False
    )
    return [{"hpo_id": item[0], "hpo_name": item[1]} for item in picked]


def sample_patient(
    rng: np.random.Generator,
    disease_rows: pd.DataFrame,
    noise_pool: list[tuple[str, str]],
    sibling_map: dict[str, list[tuple[str, str]]],
    peer_pool: dict[str, list[tuple[str, str, str]]],
    graph,
    patient_id: str,
) -> dict:
    disease_name = disease_rows["disease_name"].iloc[0]
    phenotype_rows = disease_rows[
        ["hpo_id", "hpo_name", "frequency_probability"]
    ].drop_duplicates("hpo_id")
    phenotype_rows["frequency_probability"] = phenotype_rows[
        "frequency_probability"
    ].astype(float)

    draws = rng.random(len(phenotype_rows))
    latent_selected = phenotype_rows[
        draws < phenotype_rows["frequency_probability"].to_numpy()
    ].copy()
    latent_selected = ensure_minimum_selection(
        latent_selected, phenotype_rows, MIN_TRUE_PHENOTYPES
    ).drop_duplicates("hpo_id")

    observed_true_rows, dropped_signature_rows = apply_dropout(
        rng, latent_selected
    )
    disease_hpo_ids = set(phenotype_rows["hpo_id"])

    corruptions, replaced_ids = sample_corruptions(
        rng, observed_true_rows, disease_hpo_ids, sibling_map
    )
    observed_true_rows = observed_true_rows[
        ~observed_true_rows["hpo_id"].isin(replaced_ids)
    ].copy()

    true_rows = [
        {"hpo_id": row.hpo_id, "hpo_name": row.hpo_name}
        for row in observed_true_rows.itertuples(index=False)
    ]

    used_ids = set(disease_hpo_ids) | {item["hpo_id"] for item in corruptions}
    generalized_true_rows, generalization_log = generalize_phenotypes(
        rng, true_rows, graph, used_ids
    )

    excluded_noise_ids = (
        {row["hpo_id"] for row in generalized_true_rows}
        | {item["hpo_id"] for item in corruptions}
        | disease_hpo_ids
    )
    peer_confounders = sample_peer_confounders(
        rng, disease_name, peer_pool, excluded_noise_ids
    )
    excluded_noise_ids.update(item["hpo_id"] for item in peer_confounders)
    random_noise = sample_random_noise(
        rng, disease_hpo_ids, excluded_noise_ids, noise_pool
    )

    corrupted_phenotypes = [
        {"hpo_id": item["hpo_id"], "hpo_name": item["hpo_name"]}
        for item in corruptions
    ]
    peer_phenotypes = [
        {"hpo_id": item["hpo_id"], "hpo_name": item["hpo_name"]}
        for item in peer_confounders
    ]
    all_phenotypes = (
        generalized_true_rows
        + corrupted_phenotypes
        + peer_phenotypes
        + random_noise
    )

    return {
        "patient_id": patient_id,
        "database_id": disease_rows["database_id"].iloc[0],
        "mondo_id": disease_rows["mondo_id"].iloc[0],
        "disease_name": disease_name,
        "true_diagnosis": disease_name,
        "latent_true_phenotypes": [
            {"hpo_id": row.hpo_id, "hpo_name": row.hpo_name}
            for row in latent_selected.itertuples(index=False)
        ],
        "phenotypes": generalized_true_rows,
        "signature_dropout_phenotypes": dropped_signature_rows,
        "generalization_log": generalization_log,
        "corrupted_phenotypes": corrupted_phenotypes,
        "peer_confounder_phenotypes": peer_confounders,
        "random_noise_phenotypes": random_noise,
        "noise_phenotypes": corrupted_phenotypes
        + peer_phenotypes
        + random_noise,
        "all_phenotypes": all_phenotypes,
    }


def main() -> None:
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    if "hpo_name" not in annotations.columns:
        annotations["hpo_name"] = annotations["hpo_id"]

    graph = load_graph()
    sibling_map = build_sibling_map(graph)
    peer_pool = build_peer_confounder_pool(annotations)
    rng = np.random.default_rng(SEED)
    noise_pool = list(
        annotations[["hpo_id", "hpo_name"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    patients = []
    for disease_name, disease_rows in annotations.groupby(
        "disease_name", sort=True
    ):
        for index in range(PATIENTS_PER_DISEASE):
            patient_id = f"{disease_name.lower().replace(' ', '_').replace('-', '_')}_{index + 1:02d}"
            patients.append(
                sample_patient(
                    rng,
                    disease_rows,
                    noise_pool,
                    sibling_map,
                    peer_pool,
                    graph,
                    patient_id,
                )
            )

    with SIMULATED_JSONL.open("w", encoding="utf-8") as handle:
        for patient in patients:
            handle.write(json.dumps(patient, ensure_ascii=False) + "\n")

    summary = pd.DataFrame(
        [
            {
                "patient_id": patient["patient_id"],
                "disease_name": patient["disease_name"],
                "database_id": patient["database_id"],
                "num_latent_true_phenotypes": len(
                    patient["latent_true_phenotypes"]
                ),
                "num_true_phenotypes": len(patient["phenotypes"]),
                "num_signature_dropouts": len(
                    patient["signature_dropout_phenotypes"]
                ),
                "num_generalized_phenotypes": len(
                    patient["generalization_log"]
                ),
                "num_corrupted_phenotypes": len(
                    patient["corrupted_phenotypes"]
                ),
                "num_peer_confounder_phenotypes": len(
                    patient["peer_confounder_phenotypes"]
                ),
                "num_random_noise_phenotypes": len(
                    patient["random_noise_phenotypes"]
                ),
                "num_noise_phenotypes": len(patient["noise_phenotypes"]),
                "num_total_phenotypes": len(patient["all_phenotypes"]),
            }
            for patient in patients
        ]
    )
    summary.to_csv(SIMULATED_SUMMARY, index=False)

    print(f"Saved {len(patients)} simulated patients to {SIMULATED_JSONL}")
    print(summary.groupby("disease_name")["patient_id"].count().to_string())


if __name__ == "__main__":
    main()
