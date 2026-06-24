"""File node — fetches file content / retrieval chunks from Dare."""
from handlers.base import BaseHandler


class FileHandler(BaseHandler):
    def execute(self, node, context, services, state) -> dict:
        data = node.get("data", {})
        files = data.get("files", [])
        file_ids = [f["fileId"] for f in files]
        mode = data.get("retrievalMode", "content")
        include_meta = data.get("includeMetadata", True)
        parts = []

        if mode in ("content", "both"):
            for f in files:
                content = services.dare.get_file_content(f["fileId"])
                if include_meta:
                    parts.append(f"From {f.get('name')}:\n{content}")
                else:
                    parts.append(content)

        if mode in ("embeddings", "both"):
            query = self._query_text(data, context)
            chunks = services.dare.retrieval_query(
                query=query,
                file_ids=file_ids,
                top_k=data.get("maxResults", 10),
                similarity_threshold=data.get("similarityThreshold", 0.5),
            )
            for c in chunks:
                if include_meta:
                    parts.append(f"From {c['file_name']} (score {c['score']:.2f}):\n{c['text']}")
                else:
                    parts.append(c["text"])

        return {
            "output": "\n\n".join(parts),
            "metadata": {"file_ids": file_ids, "retrieval_mode": mode},
        }

    def _query_text(self, data, context) -> str:
        if data.get("querySource") == "text_input":
            return data.get("textInput", "")
        # previous_step (default): use the most recent upstream output.
        return context[-1]["output"] if context else data.get("textInput", "")
