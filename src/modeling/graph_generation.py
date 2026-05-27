"""
Creates graph and flowchart representations of model mechanisms.
"""

from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import json
import os


def build_mechanism_graph(edges, output_path="outputs/mechanism_graph.html"):
    """
    Build an interactive mechanism graph using PyVis.

    Nodes = biological/model entities
    Edges = mechanistic relationships
    """

    from pyvis.network import Network

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


class MechanismEdge(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str
    page: str | None = None

class MechanismGraph(BaseModel):
    edges: List[MechanismEdge]

def extract_mechanism_edges_service(context: str) -> str:
    """
    Extract graph-ready source-relation-target edges from model mechanisms.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    structured_llm = llm.with_structured_output(MechanismGraph)

    prompt = f"""
    You are extracting graph-ready mechanistic model edges from a scientific modelling paper.

    Your goal is to reconstruct the mechanistic structure of the model as a graph.

    Extract only relationships explicitly used in:
    - mathematical equations
    - mechanistic model structure
    - coupling hypotheses
    - model diagrams
    - compartment diagrams
    - simulation assumptions
    - explicit model descriptions

    Do NOT extract:
    - general background biology
    - speculative mechanisms
    - literature discussion not directly implemented in the model
    - observational statements unless explicitly modeled

    Causal and directional extraction rules:
    1. Preserve directionality exactly as reported in the paper.
    Do not reverse causal, transport, regulatory,
    or compartmental relationships.

    2. Only extract relationships explicitly supported by:
    - equations
    - figure captions
    - model diagrams
    - table descriptions
    - mechanistic text
    - model assumptions

    3. Do not infer biological effects unless explicitly stated.

    4. Distinguish carefully between:
    - stimulation
    - inhibition
    - activation
    - repression
    - transport
    - transfer
    - conversion
    - degradation
    - elimination
    - production
    - consumption
    - regulation
    - feedback
    - delay
    - association/correlation

    5. If directionality, mechanism, or causal meaning is ambiguous:
    - mark as uncertain
    - include supporting evidence
    - prefer human review

    6. Prefer relationships directly supported by equations
    over qualitative narrative descriptions.

    Allowed relation types:
    - stimulates
    - inhibits
    - activates
    - represses
    - increases
    - decreases
    - transfers_to
    - transports_to
    - converts_to
    - produces
    - consumes
    - degrades
    - eliminates
    - regulates
    - delays_effect_on
    - feedback_positive
    - feedback_negative
    - associated_with
    - uncertain_relation

    For each edge return:

    - source
    - relation
    - target
    - confidence
        * explicit_equation
        * explicit_model_text
        * figure_supported
        * inferred_uncertain
    - evidence
    - page if available
    - requires_human_review (true/false)

    Graph extraction rules:
    - Return only graph-ready mechanistic edges.
    - Use consistent biological/entity names.
    - Preserve compartment names exactly when possible.
    - If an edge comes only from narrative text and is not clearly implemented in the model,
    mark requires_human_review = true.
    - If OCR/parser omitted diagrams or equations,
    avoid inventing missing edges.

    If the source text does not explicitly contain a source-relation-target relationship, do not create an edge.
    Do not convert vague biological statements into graph edges.
    Do not use "produces", "degrades", "transfers_to", or "regulates" unless the exact relationship is explicitly stated.
    If unsure, omit the edge rather than guessing.
    Prefer fewer high-confidence edges over many speculative edges.

    Context:
    {context}
    """

    result = structured_llm.invoke(prompt)
    return result.model_dump_json(indent=2)


def generate_mechanism_graph_service(edges_json: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    data = json.loads(edges_json)

    if isinstance(data, dict) and "edges" in data:
        edges = data["edges"]
    else:
        edges = data

    output_path = "outputs/mechanism_graph.html"
    build_mechanism_graph(edges, output_path=output_path)

    return f"Mechanism graph generated and saved to {output_path}"
