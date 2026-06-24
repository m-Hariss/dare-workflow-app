"""Validate an exported workflow payload and build its graph.

Accepts both the old camelCase format and the new snake_case format — all keys
are normalised to camelCase before any validation or graph construction so the
rest of the execution stack doesn't need to know which format arrived.
"""
from graph import Graph

SUPPORTED_SCHEMA_VERSIONS = {"v1"}


# ---------------------------------------------------------------------------
# Key normalisation  (snake_case → camelCase)
# ---------------------------------------------------------------------------

def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _normalize(obj):
    """Recursively convert every dict key from snake_case to camelCase."""
    if isinstance(obj, dict):
        return {_to_camel(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_workflow(payload: dict) -> Graph:
    payload = _normalize(payload)

    version = payload.get("schemaVersion")
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
