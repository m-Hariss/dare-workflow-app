"""Step node — builds the prompt, injects context/files, calls the LLM."""
from handlers.base import BaseHandler
from prompts import build_step_message


class StepHandler(BaseHandler):
    def execute(self, node, context, services, state) -> dict:
        data = node.get("data", {})
        rag  = data.get("rag", {})

        extra_files = []

        # Inline content files attached directly to this step.
        for f in rag.get("contentFiles", []):
            name = f.get("name") or str(f.get("fileId", "unknown"))
            content = services.file_store.get_content(name)
            extra_files.append(f"From {name}:\n{content}")

        # Embedding-based retrieval for files attached to this step.
        embedding_files = rag.get("embeddingFiles", [])
        if embedding_files:
            query = data.get("textInput") or (context[-1]["output"] if context else "")
            file_names = [f.get("name") or str(f.get("fileId", "unknown")) for f in embedding_files]
            chunks = services.embed.search(
                query=query,
                top_k=rag.get("maxContextSnippets", 4),
                threshold=rag.get("documentSimilarityThreshold", 0.5),
                filenames=file_names,
            )
            for c in chunks:
                extra_files.append(f"From {c['filename']} (score {c['score']:.2f}):\n{c['text']}")

        prompt  = data.get("prompt") or {}
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
