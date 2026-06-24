"""File node — reads file content or runs semantic search via ChromaDB."""
from handlers.base import BaseHandler


class FileHandler(BaseHandler):
    def execute(self, node, context, services, state) -> dict:
        data         = node.get("data", {})
        files        = data.get("files", [])
        mode         = data.get("retrievalMode", "content")
        include_meta = data.get("includeMetadata", True)
        top_k        = data.get("maxResults", 10)
        threshold    = data.get("similarityThreshold", 0.5)
        parts        = []

        file_names = [f.get("name") or str(f.get("fileId", "unknown")) for f in files]

        if mode in ("content", "both"):
            for name in file_names:
                content = services.file_store.get_content(name)
                parts.append(f"From {name}:\n{content}" if include_meta else content)

        if mode in ("embeddings", "both"):
            query = self._query_text(data, context)
            chunks = services.embed.search(
                query=query,
                top_k=top_k,
                threshold=threshold,
                filenames=file_names if file_names else None,
            )
            for c in chunks:
                label = f"From {c['filename']} (score {c['score']:.2f}):\n"
                parts.append(label + c["text"] if include_meta else c["text"])

        return {
            "output": "\n\n".join(parts),
            "metadata": {"files": file_names, "retrieval_mode": mode},
        }

    def _query_text(self, data, context) -> str:
        if data.get("querySource") == "text_input":
            return data.get("textInput", "")
        return context[-1]["output"] if context else data.get("textInput", "")
