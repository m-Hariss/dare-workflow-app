"""Generic graph execution engine.

Walks nodes in topological order, respecting router decisions (a routed target
only runs if its router selected that route). Contains no node-specific business
logic — that lives in the handlers.
"""
import logging

from handlers.registry import get_handler
from core.state import StateManager

logger = logging.getLogger(__name__)

ROUTE_HANDLE_PREFIX = "output-"


class ExecutionEngine:
    def __init__(self, graph, services):
        self.graph = graph
        self.services = services
        self.state = StateManager()

    def run(self) -> dict:
        for node_id in self.graph.topological_order():
            node = self.graph.nodes[node_id]
            node_type = node["type"]

            if not self._should_execute(node):
                self.state.mark_skipped(node_id, node_type, "branch not taken or predecessor skipped")
                logger.info("Skipped %s (%s)", node_id, node_type)
                continue

            context = self._gather_context(node)
            try:
                handler = get_handler(node_type)
                result = handler.execute(node, context, self.services, self.state)
                self.state.record(
                    node_id, node_type, "completed",
                    output=result.get("output"),
                    metadata=result.get("metadata"),
                )
                logger.info("Completed %s (%s)", node_id, node_type)
            except Exception as e:
                logger.exception("Node %s (%s) failed", node_id, node_type)
                self.state.record(node_id, node_type, "failed", error=str(e))
                break

        return self._final_result()

    # ------------------------------------------------------------------

    def _should_execute(self, node) -> bool:
        node_id = node["id"]
        if node_id == self.graph.entry_node:
            return True

        incoming = self.graph.in_edges.get(node_id, [])
        if not incoming:
            return True  # standalone node

        routed, plain = [], []
        for edge in incoming:
            src_node = self.graph.nodes.get(edge["source"])
            handle = edge.get("sourceHandle") or ""
            if src_node and src_node["type"] == "router" and handle.startswith(ROUTE_HANDLE_PREFIX):
                routed.append(edge)
            else:
                plain.append(edge)

        # Routed edge: run only if the router picked this route.
        for edge in routed:
            router_res = self.state.results.get(edge["source"])
            if not router_res or router_res["status"] != "completed":
                continue
            selected = router_res["metadata"].get("selected_route")
            route = edge["sourceHandle"][len(ROUTE_HANDLE_PREFIX):]
            if selected == route:
                return True

        # Plain edge: run if any predecessor completed.
        for edge in plain:
            if self.state.completed(edge["source"]):
                return True

        return False

    def _gather_context(self, node) -> list:
        context = []
        edges = sorted(self.graph.in_edges.get(node["id"], []), key=lambda e: e["source"])
        for edge in edges:
            res = self.state.results.get(edge["source"])
            if res and res["status"] == "completed" and res.get("output"):
                context.append({
                    "node_id": edge["source"],
                    "type": res["type"],
                    "output": res["output"],
                })
        return context

    def _final_result(self) -> dict:
        any_failed = any(r["status"] == "failed" for r in self.state.results.values())
        return {
            "workflow_id": self.graph.payload.get("workflowId"),
            "status": "failed" if any_failed else "completed",
            "entry_node": self.graph.entry_node,
            "node_results": self.state.results,
            "history": self.state.history,
            "final_output": self._terminal_outputs(),
        }

    def _terminal_outputs(self) -> dict:
        terminals = [nid for nid in self.graph.nodes if not self.graph.out_edges.get(nid)]
        return {
            nid: self.state.results[nid]["output"]
            for nid in terminals
            if self.state.completed(nid)
        }
