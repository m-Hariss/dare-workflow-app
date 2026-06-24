"""Base handler contract.

Each handler implements execute() and returns a dict:
    {"output": str, "metadata": dict}
"""


class BaseHandler:
    def execute(self, node, context, services, state) -> dict:
        raise NotImplementedError
