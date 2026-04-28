from __future__ import annotations
import json
import random
import sys
from pathlib import Path
import graphviz
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.diagnose import (  # noqa: E402
    ALL_MODES,
    Artefacts,
    diagnose,
    extract_phenotypes_with_llm,
    load_artefacts,
    load_patients,
)


st.set_page_config(
    page_title="Agentic RAG · Rare Disease Differential",
    layout="wide",
)


@st.cache_resource(
    show_spinner="Loading phenotype graph, disease profiles, and search index…"
)
def get_artefacts() -> Artefacts:
    return load_artefacts()


@st.cache_resource(show_spinner="Loading simulated patient cohort…")
def get_simulated_patients() -> list[dict]:
    return load_patients()


@st.cache_data
def get_phenotype_options(_artefacts: Artefacts) -> list[tuple[str, str]]:
    """(label, hpo_id) pairs for the multiselect, sorted by label."""
    options: dict[str, str] = {}
    for profile in _artefacts.profiles.values():
        for row in profile["phenotype_rows"]:
            label = f"{row['hpo_name']} ({row['hpo_id']})"
            options[label] = row["hpo_id"]
    for node_id, data in _artefacts.graph.nodes(data=True):
        if data.get("node_type") != "phenotype":
            continue
        name = data.get("name")
        if not name:
            continue
        label = f"{name} ({node_id})"
        options.setdefault(label, node_id)
    return sorted(options.items(), key=lambda kv: kv[0].lower())


MODE_LABELS = {
    "static": "Symptoms Only",
    "rag": "Symptoms + Text Evidence",
    "rag_graph": "Symptoms + Text + Relationship Hints",
    "agentic": "Guided Tool Use",
}

MODE_SHORT_HELP = {
    "static": "Uses only the selected symptoms and the disease list.",
    "rag": "Adds retrieved reference text before ranking the diseases.",
    "rag_graph": "Adds relationship hints on top of the retrieved text.",
    "agentic": "Lets the model choose which supporting tools to use before ranking the diseases.",
}


def render_diagnosis_card(result: dict) -> None:
    diagnosis = result["final_diagnosis"]
    confidence = result["confidence"]
    agrees = result["heuristic_agrees_with_llm"]
    cols = st.columns([3, 1, 1])
    with cols[0]:
        st.markdown(f"### Predicted diagnosis: **{diagnosis}**")
    with cols[1]:
        st.metric("Confidence", f"{confidence:.0%}")
    with cols[2]:
        agrees_str = "agrees" if agrees else "disagrees"
        delta_color = "normal" if agrees else "inverse"
        st.metric(
            "Heuristic top-1", agrees_str, delta=None, delta_color=delta_color
        )


def render_ranked_candidates(result: dict) -> None:
    ranked = result["ranked_diseases"]
    if not ranked:
        st.info("No ranked diseases were returned.")
        return
    for index, entry in enumerate(ranked, start=1):
        header = f"#{index} · {entry['disease_name']} · {entry['database_id']}"
        with st.expander(header, expanded=(index == 1)):
            st.markdown(
                f"**Why:** {entry['reason'] or '_no reason returned_'}"
            )

            tabs = st.tabs(
                [
                    "Matched phenotypes",
                    "Text evidence",
                    "Relationship hints",
                    "Disease profile",
                ]
            )

            with tabs[0]:
                matches = entry["exact_phenotype_matches"]
                if matches:
                    st.dataframe(
                        pd.DataFrame(matches),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption(
                        "No exact symptom matches were found for this disease."
                    )

            with tabs[1]:
                evidence = entry["evidence"]
                if not evidence:
                    st.caption(
                        "No text evidence was retrieved for this disease in this run."
                    )
                for snippet in evidence:
                    st.markdown(f"**{snippet.get('title') or '(untitled)'}**")
                    st.caption(
                        f"source: `{snippet.get('source', '?')}` · distance: "
                        f"{snippet.get('distance', '?'):.3f}"
                        if isinstance(snippet.get("distance"), (int, float))
                        else f"source: `{snippet.get('source', '?')}`"
                    )
                    document = snippet.get("document", "")
                    st.markdown(
                        document
                        if len(document) < 1200
                        else document[:1200] + " …"
                    )
                    st.divider()

            with tabs[2]:
                bridges = entry["ontology_bridges"]
                if not bridges:
                    st.caption(
                        "No relationship hints were found for this disease."
                    )
                else:
                    bridge_df = pd.DataFrame(
                        [
                            {
                                "observed phenotype": b["observed_hpo_name"],
                                "→ ancestor": b["bridge_ancestor_name"],
                                "← disease phenotype": b["disease_hpo_name"],
                                "distance": b["distance"],
                                "closeness": b["closeness"],
                            }
                            for b in bridges
                        ]
                    )
                    st.dataframe(
                        bridge_df, use_container_width=True, hide_index=True
                    )

            with tabs[3]:
                st.caption(
                    f"Example symptom profile · disease code: `{entry['mondo_id']}`"
                )
                st.dataframe(
                    pd.DataFrame(entry["profile_summary"]),
                    use_container_width=True,
                    hide_index=True,
                )


def render_trace(trace: list[dict]) -> None:
    if not trace:
        st.info("No run details were recorded.")
        return
    for index, event in enumerate(trace, start=1):
        with st.container(border=True):
            top = st.columns([4, 1])
            with top[0]:
                st.markdown(f"**Step {index}: {event['label']}**")
                st.caption(f"`{event['tool']}`")
            with top[1]:
                st.metric("Duration", f"{event['duration_ms']:.0f} ms")
            cols = st.columns(2)
            with cols[0]:
                st.caption("input")
                st.code(
                    json.dumps(event["input_summary"], indent=2),
                    language="json",
                )
            with cols[1]:
                st.caption("output")
                st.code(
                    json.dumps(event["output_summary"], indent=2),
                    language="json",
                )


def render_bridge_graph(result: dict) -> None:
    if result["mode"] not in {"rag_graph", "agentic"}:
        st.info(
            "This graph is only available in **With Text + Relationship Hints** or "
            "**Agentic Tool Use** mode. "
            "Switch the mode in the sidebar and run again to see it."
        )
        return
    ranked = result["ranked_diseases"]
    if not ranked:
        st.info("No ranked diseases yet.")
        return
    target = ranked[0]
    bridges = target["ontology_bridges"]
    if not bridges:
        st.info(
            f"No relationship hints connect the observed phenotypes to "
            f"**{target['disease_name']}**."
        )
        return

    def safe_node_id(prefix: str, raw_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(raw_id))
        return f"{prefix}_{cleaned}"

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", splines="true", bgcolor="transparent")
    dot.attr("node", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="9", color="#888888")

    with dot.subgraph(name="cluster_observed") as sub:
        sub.attr(label="observed phenotypes", style="dashed", color="#1f77b4")
        observed_nodes = {}
        for bridge in bridges:
            observed_nodes[bridge["observed_hpo_id"]] = bridge[
                "observed_hpo_name"
            ]
        for observed_hpo_id, observed_hpo_name in observed_nodes.items():
            sub.node(
                safe_node_id("obs", observed_hpo_id),
                observed_hpo_name,
                shape="box",
                style="filled",
                fillcolor="#dce6f4",
            )

    with dot.subgraph(name="cluster_disease") as sub:
        sub.attr(
            label=f"{target['disease_name']} phenotypes",
            style="dashed",
            color="#2ca02c",
        )
        disease_nodes = {}
        for bridge in bridges:
            disease_nodes[bridge["disease_hpo_id"]] = bridge[
                "disease_hpo_name"
            ]
        for disease_hpo_id, disease_hpo_name in disease_nodes.items():
            sub.node(
                safe_node_id("dx", disease_hpo_id),
                disease_hpo_name,
                shape="box",
                style="filled",
                fillcolor="#dceedc",
            )

    ancestor_nodes = {}
    for bridge in bridges:
        ancestor_nodes[bridge["bridge_ancestor_id"]] = bridge[
            "bridge_ancestor_name"
        ]
    for ancestor_hpo_id, ancestor_name in ancestor_nodes.items():
        ancestor_node = safe_node_id("anc", ancestor_hpo_id)
        dot.node(
            ancestor_node,
            ancestor_name,
            shape="ellipse",
            style="filled",
            fillcolor="#fff4c2",
        )

    for bridge in bridges:
        observed_node = safe_node_id("obs", bridge["observed_hpo_id"])
        ancestor_node = safe_node_id("anc", bridge["bridge_ancestor_id"])
        disease_node = safe_node_id("dx", bridge["disease_hpo_id"])
        dot.edge(
            observed_node,
            ancestor_node,
            label=f"d={bridge['distance']}",
        )
        dot.edge(
            ancestor_node,
            disease_node,
            label=f"closeness={bridge['closeness']}",
        )

    st.graphviz_chart(dot, use_container_width=True)
    st.caption(
        "Yellow nodes show shared parent concepts between an observed phenotype "
        "(left, blue) and a phenotype linked to the predicted disease (right, green). "
        "This helps explain matches even when the terms are not identical."
    )


def main() -> None:
    st.title("Rare Disease Differential Diagnosis")
    st.caption(
        "Compare several diagnosis modes on simulated rare-disease cases."
    )

    try:
        artefacts = get_artefacts()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    patients = get_simulated_patients()
    phenotype_options = get_phenotype_options(artefacts)

    with st.sidebar:
        st.header("Run config")
        mode = st.radio(
            "Diagnosis mode",
            options=list(ALL_MODES),
            format_func=lambda m: MODE_LABELS[m],
            index=2,
        )
        st.caption(MODE_SHORT_HELP[mode])
        top_k = st.slider(
            "Top-K to display", min_value=3, max_value=10, value=5
        )
        st.divider()
        st.caption(
            f"**Diseases in the app:** {len(artefacts.profiles)}\n\n"
            f"**Phenotype graph:** {artefacts.graph.number_of_nodes()} nodes, "
            f"{artefacts.graph.number_of_edges()} edges\n\n"
            f"**Simulated patients available:** {len(patients)}"
        )
        st.divider()
        st.caption("Load or enter symptoms, then click **Diagnose**.")

    st.subheader("Step 1 · Choose Input")
    if "selected_phenotype_ids" not in st.session_state:
        st.session_state.selected_phenotype_ids = []

    input_tabs = st.tabs(
        ["Simulated patient", "Clinical vignette", "Manual phenotype list"]
    )

    with input_tabs[0]:
        st.caption("Loads a prepared sample case.")
        if not patients:
            st.warning(
                "No sample patients found. Run "
                "`python -m src.patients.hpoa_simulator` first."
            )
        else:
            disease_names = sorted({p["disease_name"] for p in patients})
            disease_filter = st.selectbox(
                "Filter by diagnosis",
                options=["(any)"] + disease_names,
                index=0,
            )
            if st.button("Load random simulated patient"):
                pool = (
                    patients
                    if disease_filter == "(any)"
                    else [
                        p
                        for p in patients
                        if p["disease_name"] == disease_filter
                    ]
                )
                if pool:
                    chosen = random.choice(pool)
                    st.session_state.selected_phenotype_ids = [
                        item["hpo_id"] for item in chosen["all_phenotypes"]
                    ]
                    st.session_state["loaded_patient"] = {
                        "patient_id": chosen["patient_id"],
                        "true_diagnosis": chosen["true_diagnosis"],
                        "num_phenotypes": len(chosen["all_phenotypes"]),
                        "num_generalized": len(
                            chosen.get("generalization_log", [])
                        ),
                        "num_peer_confounders": len(
                            chosen.get("peer_confounder_phenotypes", [])
                        ),
                    }
            if st.session_state.get("loaded_patient"):
                lp = st.session_state["loaded_patient"]
                st.success(
                    f"Loaded `{lp['patient_id']}` · diagnosis: **{lp['true_diagnosis']}** · "
                    f"{lp['num_phenotypes']} phenotypes "
                    f"({lp['num_generalized']} broadened terms, "
                    f"{lp['num_peer_confounders']} distractor symptoms)"
                )

    with input_tabs[1]:
        st.caption(
            "Paste a short clinical description and the app will try to extract symptom terms."
        )
        vignette = st.text_area(
            "Paste a clinical vignette",
            height=140,
            placeholder=(
                "e.g. 5-year-old boy with progressive muscle weakness, frequent falls, "
                "calf hypertrophy, and elevated creatine kinase."
            ),
            key="vignette_text",
        )
        if st.button(
            "Extract phenotypes from vignette", disabled=not vignette.strip()
        ):
            extracted = extract_phenotypes_with_llm(
                vignette,
                artefacts.name_index,
                artefacts.graph,
                artefacts.openai_client,
            )
            if not extracted:
                st.warning(
                    "No symptom terms matched. Try mentioning specific clinical features "
                    "(e.g. *muscle weakness*, *seizure*, *cardiomyopathy*)."
                )
            else:
                st.session_state.selected_phenotype_ids = [
                    m["hpo_id"] for m in extracted
                ]
                st.success(f"Extracted {len(extracted)} symptom terms.")
                st.dataframe(
                    pd.DataFrame(extracted)[
                        ["hpo_id", "hpo_name", "matched_span"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with input_tabs[2]:
        st.caption("Select symptom terms directly.")
        labels_for_default = [
            label
            for label, hpo_id in phenotype_options
            if hpo_id in st.session_state.selected_phenotype_ids
        ]
        picked = st.multiselect(
            "Phenotype terms",
            options=[label for label, _ in phenotype_options],
            default=labels_for_default,
            help="Type to search across the phenotype terms available in this app.",
        )
        label_to_id = dict(phenotype_options)
        st.session_state.selected_phenotype_ids = [
            label_to_id[label] for label in picked
        ]

    selected_ids = st.session_state.selected_phenotype_ids
    st.subheader("Step 2 · Review Symptoms")
    if selected_ids:
        observed_phenotypes = []
        for hpo_id in selected_ids:
            name = artefacts.graph.nodes.get(hpo_id, {}).get("name", hpo_id)
            observed_phenotypes.append({"hpo_id": hpo_id, "hpo_name": name})
        st.caption(f"Selected {len(observed_phenotypes)} symptoms")
        st.dataframe(
            pd.DataFrame(observed_phenotypes),
            use_container_width=True,
            hide_index=True,
            height=180,
        )
    else:
        observed_phenotypes = []
        st.caption("No symptoms selected yet.")

    st.subheader("Step 3 · Run Diagnosis")
    run = st.button(
        "Diagnose",
        type="primary",
        disabled=not observed_phenotypes,
        use_container_width=True,
    )

    if run:
        with st.spinner(f"Running agent in `{mode}` mode…"):
            try:
                result = diagnose(
                    observed_phenotypes, mode, artefacts, top_k=top_k
                )
            except Exception as exc:
                st.error(f"Agent failed: {exc}")
                st.exception(exc)
                st.stop()
        st.session_state["last_result"] = result
        st.session_state["last_run_mode"] = mode

    result = st.session_state.get("last_result")
    if not result:
        st.divider()
        st.info(
            "Pick symptoms or load a sample case, then click **Diagnose**."
        )
        return

    st.divider()
    if result.get("mode") != mode:
        st.warning(
            f"You are currently viewing a **{result['mode']}** result, but the sidebar is set to "
            f"**{mode}**. Click **Diagnose** again to rerun in the selected mode."
        )
    else:
        st.caption(
            f"Showing result for **{MODE_LABELS[result['mode']]}**. "
            "Switch modes in the sidebar and rerun to compare."
        )
    render_diagnosis_card(result)
    st.divider()

    output_tabs = st.tabs(["Ranked diagnoses", "Trace", "Relationship graph"])
    with output_tabs[0]:
        render_ranked_candidates(result)
    with output_tabs[1]:
        render_trace(result["trace"])
    with output_tabs[2]:
        render_bridge_graph(result)


if __name__ == "__main__":
    main()
