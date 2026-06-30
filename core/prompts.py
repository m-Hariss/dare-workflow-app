"""Prompt assembly helpers.

Mirrors Dare's step message structure (<instructions>/<workflow_context>/<task>)
so the runner produces equivalent prompts to the in-Dare engine.
"""

DEFAULT_TASK_MESSAGE = "Please complete the task described."


def build_step_message(instructions, context, task, extra_files=None) -> str:
    parts = []

    if instructions:
        parts.append(f"<instructions>\n{instructions}\n</instructions>")

    if extra_files:
        parts.append("<files>\n" + "\n\n".join(extra_files) + "\n</files>")

    if context:
        blocks = []
        for c in context:
            blocks.append(
                f'<upstream_node id="{c["node_id"]}" type="{c["type"]}">\n'
                f'<output><![CDATA[{c["output"]}]]></output>\n'
                f'</upstream_node>'
            )
        parts.append("<workflow_context>\n" + "\n".join(blocks) + "\n</workflow_context>")

    parts.append(f"<task>\n{task or DEFAULT_TASK_MESSAGE}\n</task>")
    return "\n\n".join(parts)


def build_router_message(instructions, routes, upstream, task) -> str:
    route_names = [r["name"] for r in routes]
    lines = []
    if instructions:
        lines.append(instructions)
    lines.append("\nYou are a routing decision maker. Choose exactly ONE route.")
    lines.append("Available routes:")
    for r in routes:
        lines.append(f"- {r['name']}: {r.get('description', '')}")
    if upstream:
        lines.append(f"\nContext from previous step:\n{upstream}")
    if task:
        lines.append(f"\nAdditional input: {task}")
    lines.append(
        "\nRespond with EXACTLY one of these values and nothing else: "
        + ", ".join(route_names)
    )
    return "\n".join(lines)
