from handlers.base import BaseHandler


class OutputHandler(BaseHandler):
    """Display node — passes the upstream output through unchanged."""

    def execute(self, node, context, services, state) -> dict:
        text = context[-1]["output"] if context else ""
        return {"output": text, "metadata": {"label": node.get("data", {}).get("label")}}
