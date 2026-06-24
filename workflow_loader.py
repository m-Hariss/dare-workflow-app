"""Validate an exported workflow payload and build its graph."""
from graph import Graph

SUPPORTED_SCHEMA_VERSIONS = {"v1"}


def load_workflow(payload: dict) -> Graph:
    version = payload.get("schemaVersion") or payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version: {version!r} "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    if not payload.get("entryNode"):
        raise ValueError("Export is missing 'entryNode'")
    if not payload.get("nodes"):
        raise ValueError("Export has no nodes")

    graph = Graph(payload)
    if graph.entry_node not in graph.nodes:
        raise ValueError(f"entryNode {graph.entry_node!r} not found in nodes")

    return graph
