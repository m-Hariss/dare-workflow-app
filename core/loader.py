"""Validate an exported workflow payload and normalise it to a canonical shape.

Dare has shipped two export schema versions:

  v1  — files are *baked into* the export (data.files[].name) and retrieval
        settings are flat keys on the node data. Designed to run inside Dare.

  v2  — files are *declared inputs* the end user supplies locally
        (data.fileInputs[] + top-level requiredInputs[]), and retrieval settings
        live in a nested data.retrieval object. Designed to run detached, in this
        Syftbox app.

To keep the handlers/engine simple, every export — regardless of version — is
converted here into ONE canonical internal shape (the v1-style layout the
handlers already understand):

  file node data:  files=[{name, slot, label, usage, key}], retrievalMode,
                   similarityThreshold, maxResults, querySource, includeMetadata
  step node data:  rag={contentFiles, embeddingFiles, ...}

For v2, file inputs carry no filename (the user brings their own file), so each
input is given a stable synthetic slot id — "<nodeId>" for a single input, or
"<nodeId>::<key>" when a node declares several. That slot id is what the upload
slot, the embedding index, and the file handler all key on.
"""
from core.graph import Graph

SUPPORTED_SCHEMA_VERSIONS = {"v1", "v2"}


# ---------------------------------------------------------------------------
# Key normalisation  (snake_case → camelCase)
# ---------------------------------------------------------------------------

def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _normalize_keys(obj):
    """Recursively convert every dict key from snake_case to camelCase."""
    if isinstance(obj, dict):
        return {_to_camel(k): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# v2 → canonical adaptation
# ---------------------------------------------------------------------------

def _slot_id(node_id: str, key: str, single: bool) -> str:
    """Stable identifier for a v2 file input — what the upload slot keys on."""
    return node_id if single else f"{node_id}::{key}"


def _usage_needs_embeddings(usage: str) -> bool:
    return usage in ("embed", "embed_and_retrieve", "embeddings")


def _adapt_v2_file_node(node: dict):
    data        = node.get("data", {})
    node_id     = node.get("id", "")
    label       = data.get("label", "File")
    file_inputs = data.get("fileInputs", [])
    retrieval   = data.get("retrieval", {})

    single = len(file_inputs) <= 1
    data["files"] = [
        {
            "name":  _slot_id(node_id, inp.get("key", "files"), single),
            "slot":  True,
            "label": label,
            "usage": inp.get("usage", "retrieve"),
            "key":   inp.get("key", "files"),
        }
        for inp in file_inputs
    ]

    # Flatten the nested retrieval object onto the node data (canonical layout).
    data["retrievalMode"]       = retrieval.get("mode", "content")
    data["similarityThreshold"] = retrieval.get("similarityThreshold", 0.5)
    data["maxResults"]          = retrieval.get("maxResults", 10)
    data["querySource"]         = retrieval.get("querySource", "previous_step")
    data["includeMetadata"]     = retrieval.get("includeMetadata", True)
    node["data"] = data


def _adapt_v2_step_node(node: dict):
    data        = node.get("data", {})
    node_id     = node.get("id", "")
    file_inputs = data.get("fileInputs", [])

    # v2 steps may declare their own file inputs; split them into the canonical
    # rag content/embedding buckets by declared usage.
    if "rag" not in data:
        single = len(file_inputs) <= 1
        content_files, embedding_files = [], []
        for inp in file_inputs:
            ref = {
                "name":  _slot_id(node_id, inp.get("key", "files"), single),
                "slot":  True,
                "label": data.get("label", "File"),
                "key":   inp.get("key", "files"),
            }
            if _usage_needs_embeddings(inp.get("usage", "")):
                embedding_files.append(ref)
            else:
                content_files.append(ref)
        data["rag"] = {
            "contentFiles":               content_files,
            "embeddingFiles":             embedding_files,
            "maxContextSnippets":         data.get("maxContextSnippets", 4),
            "documentSimilarityThreshold": data.get("documentSimilarityThreshold", 0.5),
        }
    node["data"] = data


def _adapt_v2(payload: dict) -> dict:
    for node in payload.get("nodes", []):
        ntype = node.get("type")
        if ntype == "file":
            _adapt_v2_file_node(node)
        elif ntype in ("step", "router"):
            _adapt_v2_step_node(node)
    return payload


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_export(payload: dict) -> dict:
    """Validate the schema version and return a canonical-shape payload.

    Shared by the run path (load_workflow) and the dashboard parser so both see
    exactly one internal shape. Raises ValueError on an unsupported version.
    """
    payload = _normalize_keys(payload)

    version = payload.get("schemaVersion")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version: {version!r} "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    if version == "v2":
        payload = _adapt_v2(payload)
    return payload


def load_workflow(payload: dict) -> Graph:
    payload = normalize_export(payload)

    if not payload.get("entryNode"):
        raise ValueError("Export is missing 'entryNode'")
    if not payload.get("nodes"):
        raise ValueError("Export has no nodes")

    graph = Graph(payload)
    if graph.entry_node not in graph.nodes:
        raise ValueError(f"entryNode {graph.entry_node!r} not found in nodes")

    return graph
