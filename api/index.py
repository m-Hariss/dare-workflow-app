"""Embedding / indexing endpoints."""
from deps import key_store, make_embed_pipeline
from file_store import FileStore


def register(app):
    @app.post("/index", tags=["syftbox"])
    def index_files():
        folder = key_store.get("files_folder")
        file_store = FileStore(folder)
        if not file_store.is_configured():
            return {"error": "No files folder configured."}
        files = file_store.list_files()
        if not files:
            return {"error": "Files folder is empty."}

        pipeline = make_embed_pipeline()
        indexed, failed = [], []
        for f in files:
            try:
                content = file_store.get_content(f["name"])
                n = pipeline.index_file(f["name"], content)
                indexed.append({"name": f["name"], "chunks": n})
            except Exception as e:
                failed.append({"name": f["name"], "error": str(e)})

        return {"indexed": indexed, "failed": failed,
                "total_chunks": sum(i["chunks"] for i in indexed)}

    @app.get("/index/status", tags=["syftbox"])
    def index_status():
        return make_embed_pipeline().get_status()
