"""Step node — builds the prompt, injects context/files, calls the LLM."""
from handlers.base import BaseHandler
from prompts import build_step_message


class StepHandler(BaseHandler):
    def execute(self, node, context, services, state) -> dict:
        data = node.get("data", {})
        rag = data.get("rag", {})

        # Step-level files (separate from File nodes).
        extra_files = []
        for f in rag.get("contentFiles", []):
            content = services.dare.get_file_content(f["fileId"])
            extra_files.append(f"From {f.get('name')}:\n{content}")

        embedding_files = rag.get("embeddingFiles", [])
        if embedding_files:
            query = data.get("textInput") or (context[-1]["output"] if context else "")
            chunks = services.dare.retrieval_query(
                query=query,
                file_ids=[f["fileId"] for f in embedding_files],
                top_k=rag.get("maxContextSnippets", 4),
                similarity_threshold=rag.get("documentSimilarityThreshold", 0.5),
            )
            for c in chunks:
                extra_files.append(f"From {c['file_name']}:\n{c['text']}")

        prompt = data.get("prompt") or {}
        message = build_step_message(
            instructions=prompt.get("content", ""),
            context=context if data.get("usePreviousContext", True) else [],
            task=data.get("textInput", ""),
            extra_files=extra_files,
        )

        llm = data.get("llm") or {}
        if not llm.get("identifier"):
            raise RuntimeError(f"Step node {node['id']} has no LLM configured")

        generation = data.get("generation", {})
        text, usage = services.llm.complete(
            llm=llm,
            message=message,
            max_tokens=generation.get("maxTokens"),
            temperature=generation.get("temperature"),
            web_search=data.get("enableWebSearch", False),
        )
        return {"output": text, "metadata": {"token_usage": usage}}
