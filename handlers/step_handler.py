"""Step node — builds the prompt, injects context/files, calls the LLM."""
import logging

from handlers.base import BaseHandler
from core.prompts import build_step_message

logger = logging.getLogger(__name__)


class StepHandler(BaseHandler):
    def execute(self, node, context, services, state) -> dict:
        data = node.get("data", {})
        rag  = data.get("rag", {})
        node_id = node.get("id", "?")

        extra_files = []
        rag_metadata = {"query": None, "chunks_retrieved": 0, "chunks": []}

        # Inline content files attached directly to this step.
        for f in rag.get("contentFiles", []):
            name = f.get("name") or str(f.get("fileId", "unknown"))
            content = services.file_store.get_content(name)
            extra_files.append(f"From {name}:\n{content}")
            logger.info("[step:%s] content file injected  name=%s  chars=%d", node_id, name, len(content))

        # Embedding-based retrieval for files attached to this step.
        embedding_files = rag.get("embeddingFiles", [])
        if embedding_files:
            query = (data.get("textInput")
                     or (context[-1]["output"] if context else "")
                     or (data.get("prompt") or {}).get("content", ""))
            if query:
                file_names = [f.get("name") or str(f.get("fileId", "unknown")) for f in embedding_files]
                logger.info("[step:%s] RAG search  query_len=%d  files=%s  top_k=%s  threshold=%s",
                            node_id, len(query), file_names,
                            rag.get("maxContextSnippets", 4),
                            rag.get("documentSimilarityThreshold", 0.5))
                chunks = services.embed.search(
                    query=query,
                    top_k=rag.get("maxContextSnippets", 4),
                    threshold=rag.get("documentSimilarityThreshold", 0.5),
                    filenames=file_names,
                )
                logger.info("[step:%s] RAG retrieved %d chunk(s)  scores=%s",
                            node_id, len(chunks),
                            [round(c["score"], 3) for c in chunks])
                rag_metadata = {
                    "query": query[:120],
                    "chunks_retrieved": len(chunks),
                    "chunks": [{"filename": c["filename"], "score": c["score"],
                                "preview": c["text"][:80]} for c in chunks],
                }
                for c in chunks:
                    extra_files.append(f"From {c['filename']} (score {c['score']:.2f}):\n{c['text']}")
            else:
                logger.warning("[step:%s] RAG skipped — no query (no textInput and no prior output)", node_id)
        else:
            logger.info("[step:%s] no embeddingFiles configured", node_id)

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
        return {"output": text, "metadata": {"token_usage": usage, "rag": rag_metadata}}
