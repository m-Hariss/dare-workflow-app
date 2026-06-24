"""Tracks runtime execution state: per-node results, skips, and history."""


class StateManager:
    def __init__(self):
        self.results = {}        # node_id -> result dict
        self.skipped = set()     # node_ids skipped
        self.history = []        # ordered list of {node_id, status}

    def record(self, node_id, node_type, status, output=None, metadata=None, error=None):
        entry = {
            "node_id": node_id,
            "type": node_type,
            "status": status,
            "output": output,
            "metadata": metadata or {},
            "error": error,
        }
        self.results[node_id] = entry
        self.history.append({"node_id": node_id, "status": status})
        return entry

    def mark_skipped(self, node_id, node_type, reason=""):
        self.skipped.add(node_id)
        return self.record(node_id, node_type, "skipped", metadata={"reason": reason})

    def completed(self, node_id) -> bool:
        res = self.results.get(node_id)
        return bool(res and res["status"] == "completed")

    def get_output(self, node_id):
        res = self.results.get(node_id)
        return res["output"] if res else None
