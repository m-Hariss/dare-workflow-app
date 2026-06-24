"""Maps runtime node types to their handlers."""
from handlers.file_handler import FileHandler
from handlers.output_handler import OutputHandler
from handlers.router_handler import RouterHandler
from handlers.start_handler import StartHandler
from handlers.step_handler import StepHandler

_HANDLERS = {
    "start": StartHandler(),
    "step": StepHandler(),
    "file": FileHandler(),
    "router": RouterHandler(),
    "output": OutputHandler(),
}


def get_handler(node_type: str):
    handler = _HANDLERS.get(node_type)
    if handler is None:
        raise ValueError(f"No handler registered for node type: {node_type!r}")
    return handler
