"""In-memory graph representation of an exported Dare workflow.

The exported JSON uses camelCase keys (Dare renders responses in camelCase).
Standard library only.
"""
from collections import defaultdict, deque


class Graph:
    def __init__(self, payload: dict):
        self.payload = payload
        self.entry_node = payload.get("entryNode")
        self.nodes = {n["id"]: n for n in payload.get("nodes", [])}
        self.edges = payload.get("edges", [])

        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        for edge in self.edges:
            if edge["source"] in self.nodes:
                self.out_edges[edge["source"]].append(edge)
            if edge["target"] in self.nodes:
                self.in_edges[edge["target"]].append(edge)

    def topological_order(self) -> list:
        """Kahn's algorithm with stable (sorted) tie-breaking for determinism."""
        indeg = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge["target"] in indeg and edge["source"] in self.nodes:
                indeg[edge["target"]] += 1

        queue = deque(sorted(nid for nid, d in indeg.items() if d == 0))
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for edge in sorted(self.out_edges[nid], key=lambda e: e["target"]):
                target = edge["target"]
                if target in indeg:
                    indeg[target] -= 1
                    if indeg[target] == 0:
                        queue.append(target)

        # Safety: append any nodes left out by cycles so nothing is dropped.
        for nid in self.nodes:
            if nid not in order:
                order.append(nid)
        return order
