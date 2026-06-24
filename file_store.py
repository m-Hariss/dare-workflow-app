"""Local file store — reads files directly from a user-configured folder.

No copying or uploading. The user points to an existing folder on their machine;
the runner reads files from it by filename at execution time.

Text decoding: files are read as UTF-8 (replacement chars for undecodable bytes)
so their content can be injected into LLM prompts.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".exe", ".dll", ".so", ".dylib",
}


class FileStore:
    def __init__(self, folder: str | Path | None = None):
        self.folder = Path(folder) if folder else None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_content(self, filename: str) -> str:
        """Return the text content of a file in the configured folder."""
        if not self.folder:
            raise RuntimeError(
                "No files folder configured. Set it in the dashboard under 'Files Folder'."
            )
        if not self.folder.exists():
            raise RuntimeError(
                f"Files folder {str(self.folder)!r} does not exist. "
                "Check the path in the dashboard."
            )
        path = self.folder / Path(filename).name
        if not path.exists():
            raise FileNotFoundError(
                f"File {filename!r} not found in {self.folder}. "
                "Make sure the file is in your configured files folder."
            )
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            raise ValueError(
                f"File {filename!r} is a binary file and cannot be used as LLM context."
            )
        return path.read_text(encoding="utf-8", errors="replace")

    # ------------------------------------------------------------------
    # List (read-only view of the folder)
    # ------------------------------------------------------------------

    def list_files(self) -> list[dict]:
        if not self.folder or not self.folder.exists():
            return []
        return sorted(
            [
                {"name": p.name, "size": p.stat().st_size}
                for p in self.folder.iterdir()
                if p.is_file() and not p.name.startswith(".")
            ],
            key=lambda f: f["name"].lower(),
        )

    def is_configured(self) -> bool:
        return self.folder is not None and self.folder.exists()
