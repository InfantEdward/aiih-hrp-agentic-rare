import json
import pickle
from pathlib import Path
import networkx as nx
import pandas as pd
import pronto


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPO_PATH = DATA_DIR / "hpo" / "hp.obo"
ANNOTATIONS_PATH = OUTPUT_DIR / "locked_hpoa_annotations.csv"
GRAPH_PATH = OUTPUT_DIR / "locked_disease_graph.pkl"
STATS_PATH = OUTPUT_DIR / "locked_disease_graph_stats.json"


def main() -> None:
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    hpo = pronto.Ontology(str(HPO_PATH))
    graph = nx.DiGraph()

    phenotype_ids = set(annotations["hpo_id"])
    ancestor_ids = set()
    for hpo_id in phenotype_ids:
        try:
            term = hpo[hpo_id]
        except KeyError:
            continue
        ancestor_ids.update(
            parent.id for parent in term.superclasses(with_self=False)
        )

    all_hpo_ids = phenotype_ids | ancestor_ids

    for hpo_id in all_hpo_ids:
        try:
            term = hpo[hpo_id]
        except KeyError:
            continue
        graph.add_node(
            hpo_id,
            node_type="phenotype",
            name=term.name,
        )
        for parent in term.superclasses(distance=1, with_self=False):
            if parent.id == hpo_id:
                continue
            graph.add_edge(hpo_id, parent.id, relation="is_a")

    for row in annotations.itertuples(index=False):
        disease_node = row.database_id
        graph.add_node(
            disease_node,
            node_type="disease",
            name=row.disease_name,
            mondo_id=row.mondo_id,
        )
        graph.add_edge(
            disease_node,
            row.hpo_id,
            relation="has_phenotype",
            probability=float(row.frequency_probability),
            evidence=row.evidence,
            reference=row.reference,
        )

    with GRAPH_PATH.open("wb") as handle:
        pickle.dump(graph, handle)

    stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_diseases": int(
            sum(
                1
                for _, data in graph.nodes(data=True)
                if data.get("node_type") == "disease"
            )
        ),
        "num_phenotypes": int(
            sum(
                1
                for _, data in graph.nodes(data=True)
                if data.get("node_type") == "phenotype"
            )
        ),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved graph to {GRAPH_PATH}")


if __name__ == "__main__":
    main()
