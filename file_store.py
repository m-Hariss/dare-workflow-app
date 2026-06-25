"""Local file store — reads files directly from a user-configured folder.

No copying or uploading. The user points to an existing folder on their machine;
the runner reads files from it by filename at execution time.

file_map: maps workflow slot names (the Dare filenames) to actual local paths.
Slot files take priority over the folder — the folder is a fallback.

Text decoding: UTF-8 for plain text; python-docx for .docx; binary files rejected.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".exe", ".dll", ".so", ".dylib",
    ".xlsx", ".xls", ".pptx", ".ppt",
}


class FileStore:
    def __init__(self, folder: str | Path | None = None, file_map: dict | None = None):
        self.folder = Path(folder) if folder else None
        # {slot_name: str(path)} — uploaded replacements for workflow file references
        self._file_map: dict[str, str] = file_map or {}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_content(self, filename: str) -> str:
        """Return text content for a filename.

        Lookup order:
        1. Uploaded slot file (keyed by the workflow's original filename)
        2. Configured files folder (by filename)
        """
        # 1. Check slot map (user uploaded a replacement)
        if filename in self._file_map:
            path = Path(self._file_map[filename])
            if path.exists():
                return self._read_path(path, filename)
            logger.warning("Slot file missing on disk: %s — falling through to folder", path)

        # 2. Fall back to configured folder
        if not self.folder:
            raise RuntimeError(
                f"No uploaded file for '{filename}' and no files folder configured. "
                "Upload a file for this slot in step 3."
            )
        if not self.folder.exists():
            raise RuntimeError(
                f"Files folder {str(self.folder)!r} does not exist. "
                "Check the path in the dashboard."
            )
        path = self.folder / Path(filename).name
        if not path.exists():
            raise FileNotFoundError(
                f"File '{filename}' not found. Upload it in step 3 or add it to your files folder."
            )
        return self._read_path(path, filename)

    def _read_path(self, path: Path, label: str) -> str:
        ext = path.suffix.lower()
        if ext in _BINARY_EXTENSIONS:
            raise ValueError(
                f"File '{label}' is a binary format and cannot be used as LLM context."
            )
        if ext == ".docx":
            return self._read_docx(path)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_docx(path: Path) -> str:
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError(
                "python-docx is required to read .docx files. "
                "Run: pip install python-docx"
            )
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

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
