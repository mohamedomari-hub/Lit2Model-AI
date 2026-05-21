import os
import json
import re
from pyvis.network import Network


def extract_edges_from_agent_result(agent_result_path: str):
    """
    Try to extract graph-ready mechanism edges from agent_result.md.

    Expected edge format:
    {
      "source": "...",
      "relation": "...",
      "target": "...",
      "evidence": "...",
      "page": "..."
    }

    If no valid JSON is found, return an empty list.
    """

    if not os.path.exists(agent_result_path):
        raise FileNotFoundError(f"File not found: {agent_result_path}")

    with open(agent_result_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Find JSON-like blocks
    json_blocks = re.findall(r"\{[\s\S]*?\}", text)

    edges = []

    for block in json_blocks:
        try:
            data = json.loads(block)

            if all(key in data for key in ["source", "relation", "target"]):
                edges.append(data)

        except json.JSONDecodeError:
            continue

    return edges


def build_mechanism_graph(edges, output_path="outputs/mechanism_graph.html"):
    """
    Build an interactive mechanism graph using PyVis.

    Nodes = biological/model entities
    Edges = mechanistic relationships
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False
    )

    net.force_atlas_2based()

    for edge in edges:
        source = edge.get("source", "Unknown source")
        target = edge.get("target", "Unknown target")
        relation = edge.get("relation", "related_to")
        evidence = edge.get("evidence", "")
        page = edge.get("page", "")

        net.add_node(
            source,
            label=source,
            title=source
        )

        net.add_node(
            target,
            label=target,
            title=target
        )

        edge_title = f"Relation: {relation}<br>Page: {page}<br>Evidence: {evidence}"

        net.add_edge(
            source,
            target,
            label=relation,
            title=edge_title
        )

    net.write_html(output_path)

    return output_path


def create_demo_edges():
    """
    Fallback demo edges for testing the graph generator.
    """

    return [
        {
            "source": "Dexamethasone",
            "relation": "transfers_to",
            "target": "Central compartment",
            "evidence": "Dexamethasone enters the central compartment after intramuscular administration.",
            "page": "5"
        },
        {
            "source": "Central compartment",
            "relation": "delays_effect_on",
            "target": "Effect compartment",
            "evidence": "The effect compartment accounts for delayed drug effects.",
            "page": "6"
        },
        {
            "source": "Dexamethasone effect",
            "relation": "stimulates",
            "target": "Glucagon secretion",
            "evidence": "Dexamethasone stimulates glucagon secretion from alpha cells.",
            "page": "9"
        },
        {
            "source": "Glucagon",
            "relation": "increases",
            "target": "Gluconeogenesis",
            "evidence": "Glucagon increases hepatic gluconeogenesis.",
            "page": "9"
        },
        {
            "source": "Dexamethasone effect",
            "relation": "inhibits",
            "target": "Glucose uptake",
            "evidence": "Dexamethasone reduces insulin-stimulated glucose uptake.",
            "page": "9"
        },
        {
            "source": "Glucose",
            "relation": "increases",
            "target": "Insulin",
            "evidence": "Increased glucose concentration leads to increased insulin concentration.",
            "page": "13"
        }
    ]


def generate_graph_from_agent_output(
    agent_result_path="outputs/agent_result.md",
    output_path="outputs/mechanism_graph.html"
):
    """
    Main helper function.

    1. Try to extract edges from agent_result.md
    2. If no edges are found, use demo edges
    3. Generate HTML graph
    """

    edges = extract_edges_from_agent_result(agent_result_path)

    if not edges:
        print("No JSON edges found in agent result. Using demo edges for now.")
        edges = create_demo_edges()

    graph_path = build_mechanism_graph(
        edges=edges,
        output_path=output_path
    )

    print(f"Graph saved to: {graph_path}")

    return graph_path


if __name__ == "__main__":
    generate_graph_from_agent_output()