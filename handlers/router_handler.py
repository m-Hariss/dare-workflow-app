"""Router node — LLM picks one of the configured routes."""
from handlers.base import BaseHandler
from core.prompts import build_router_message


class RouterHandler(BaseHandler):
    def execute(self, node, context, services, state) -> dict:
        data = node.get("data", {})
        routes = data.get("routes", [])
        route_names = [r["name"] for r in routes]
        if not route_names:
            raise RuntimeError(f"Router node {node['id']} has no routes")

        # Headless run: a router that normally pauses for a human is auto-decided
        # from the LLM's recommendation (recorded in metadata for transparency).
        auto_validated = bool(data.get("requireHumanValidation"))

        prompt = data.get("prompt") or {}
        upstream = context[-1]["output"] if context else ""
        message = build_router_message(
            instructions=prompt.get("content", ""),
            routes=routes,
            upstream=upstream,
            task=data.get("textInput", ""),
        )

        llm = data.get("llm") or {}
        if not llm.get("identifier"):
            raise RuntimeError(f"Router node {node['id']} has no LLM configured")

        text, usage = services.llm.complete(llm=llm, message=message, max_tokens=100, temperature=0)
        selected = self._parse_route(text, route_names)

        return {
            "output": selected,
            "metadata": {
                "selected_route": selected,
                "ai_analysis": text,
                "available_routes": route_names,
                "auto_validated": auto_validated,
                "token_usage": usage,
            },
        }

    def _parse_route(self, text, route_names):
        cleaned = (text or "").strip().lower()
        for name in route_names:
            if name.lower() == cleaned:
                return name
        for name in route_names:
            if name.lower() in cleaned:
                return name
        return route_names[0]  # deterministic fallback
