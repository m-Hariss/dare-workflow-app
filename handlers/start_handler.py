from handlers.base import BaseHandler


class StartHandler(BaseHandler):
    """Workflow entry point — no-op, just begins execution."""

    def execute(self, node, context, services, state) -> dict:
        data = node.get("data", {})
        return {"output": "", "metadata": {"title": data.get("title")}}
