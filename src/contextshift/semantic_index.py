"""Optional, local semantic candidate index for selected workspace snapshots."""

import hashlib
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextshift.repository import RepositoryIndex, WorkspaceIndex

LOG = logging.getLogger(__name__)
MODEL_NAME = "Qwen3-Embedding-0.6B"


@dataclass(frozen=True, slots=True)
class SemanticProgress:
    stage: str
    completed: int = 0
    total: int = 0
    device: str = ""
    reused: int = 0


def _chunk_id(root: str, path: str, digest: str, start: int, end: int) -> str:
    return hashlib.sha256(
        f"v1\0{root}\0{path}\0{digest}\0{start}\0{end}".encode()
    ).hexdigest()


class SemanticIndex:
    """Build embeddings in a daemon thread; search only a fully synced snapshot."""

    def __init__(
        self,
        model_path: Path | None = None,
        cache_path: Path | None = None,
        on_progress: Callable[[SemanticProgress], None] | None = None,
    ):
        self.model_path = (
            model_path
            or Path(
                os.environ.get(
                    "CONTEXTSHIFT_EMBEDDING_MODEL",
                    str(Path.home() / "Models" / MODEL_NAME),
                )
            ).expanduser()
        )
        self.cache_path = (
            cache_path
            or Path(
                os.environ.get(
                    "CONTEXTSHIFT_EMBEDDING_CACHE",
                    str(Path.home() / ".cache" / "contextshift" / "semantic-v1"),
                )
            ).expanduser()
        )
        self._condition = threading.Condition()
        self._latest: RepositoryIndex | WorkspaceIndex | None = None
        self._generation = 0
        self._ready: str | None = None
        self._lookup: dict[str, tuple[str, int, int]] = {}
        self._table = None
        self._encoder = None
        self._encoder_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._on_progress = on_progress
        self._device = ""

    def _publish(
        self, progress: SemanticProgress, generation: int | None = None
    ) -> None:
        if self._on_progress is not None and (
            generation is None or self._current(generation)
        ):
            self._on_progress(progress)

    def set_progress_callback(
        self, callback: Callable[[SemanticProgress], None]
    ) -> None:
        self._on_progress = callback

    @property
    def available(self) -> bool:
        return (self.model_path / "model.safetensors").is_file()

    @property
    def device(self) -> str:
        """Device used by the encoder, empty until it actually runs."""
        return self._device

    def schedule(self, index: RepositoryIndex | WorkspaceIndex | None) -> None:
        if self._closed:
            return
        if not self.available:
            self._publish(SemanticProgress("unavailable"))
            return
        with self._condition:
            if index is not None and self._ready == index.fingerprint:
                return
            self._latest = index
            self._generation += 1
            self._ready = None
            self._lookup = {}
            self._condition.notify()
            generation = self._generation
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="contextshift-embeddings", daemon=True
                )
                self._thread.start()
        self._publish(SemanticProgress("queued" if index else "waiting"), generation)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()

    def _run(self) -> None:
        seen = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda previous=seen: self._closed or self._generation != previous
                )
                if self._closed:
                    return
                seen = self._generation
                index = self._latest
            if index is None:
                continue
            try:
                self._build(index, seen)
            except Exception:  # Optional index must never stop the GUI.
                LOG.exception("Local semantic index unavailable")
                self._publish(SemanticProgress("failed", device=self._device), seen)

    def _current(self, generation: int) -> bool:
        with self._condition:
            return not self._closed and self._generation == generation

    def _build(self, index: RepositoryIndex | WorkspaceIndex, generation: int) -> None:
        self._publish(SemanticProgress("scanning", device=self._device), generation)
        import lancedb
        import torch
        from sentence_transformers import SentenceTransformer

        projects = index.indexes if isinstance(index, WorkspaceIndex) else (index,)
        roots = "\0".join(str(project.root) for project in projects)
        weights = self.model_path / "model.safetensors"
        version = (
            f"{self.model_path}:{weights.stat().st_size}:{weights.stat().st_mtime_ns}"
            if weights.is_file()
            else str(self.model_path)
        )
        database = (
            self.cache_path
            / hashlib.sha256(f"{version}\0{roots}".encode()).hexdigest()[:24]
        )
        database.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(database))
        table = db.open_table("chunks") if "chunks" in db.list_tables().tables else None
        lookup = {}
        pending = []
        for number, project in enumerate(projects, 1):
            for path, source in project.files.items():
                for chunk in source.chunks:
                    ref = chunk.reference
                    key = _chunk_id(
                        str(project.root),
                        path,
                        source.digest,
                        ref.start_line,
                        ref.end_line,
                    )
                    lookup[key] = (f"P{number}/{path}", ref.start_line, ref.end_line)
                    pending.append((key, f"{path}\n{chunk.text[:4000]}"))
        if not self._current(generation):
            return
        self._publish(SemanticProgress("comparing", device=self._device), generation)
        existing = (
            set(table.search().select(["id"]).to_arrow().column("id").to_pylist())
            if table
            else set()
        )
        missing = [(key, text) for key, text in pending if key not in existing]
        reused = len(pending) - len(missing)
        self._publish(
            SemanticProgress("encoding", reused, len(pending), self._device, reused),
            generation,
        )
        if missing and self._encoder is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._publish(
                SemanticProgress("loading", reused, len(pending), device, reused),
                generation,
            )
            self._encoder = SentenceTransformer(
                str(self.model_path), device=device, local_files_only=True
            )
            self._device = device
        for offset in range(0, len(missing), 128):
            if not self._current(generation):
                return
            batch = missing[offset : offset + 128]
            with self._encoder_lock:
                vectors = self._encoder.encode_document(
                    [text for _, text in batch],
                    batch_size=8,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            rows = [
                {"id": key, "vector": vector.tolist()}
                for (key, _), vector in zip(batch, vectors, strict=True)
            ]
            if table is None:
                table = db.create_table("chunks", data=rows)
            else:
                table.add(rows)
            self._publish(
                SemanticProgress(
                    "encoding",
                    reused + offset + len(batch),
                    len(pending),
                    self._device,
                    reused,
                ),
                generation,
            )
        if not self._current(generation):
            return
        self._publish(
            SemanticProgress(
                "cleaning", len(pending), len(pending), self._device, reused
            ),
            generation,
        )
        stale = sorted(existing - lookup.keys())
        for offset in range(0, len(stale), 200):
            if not self._current(generation):
                return
            values = ", ".join(f"'{key}'" for key in stale[offset : offset + 200])
            table.delete(f"id IN ({values})")
        with self._condition:
            if self._generation == generation and not self._closed:
                self._table = table
                self._lookup = lookup
                self._ready = index.fingerprint
        self._publish(
            SemanticProgress("ready", len(pending), len(pending), self._device, reused),
            generation,
        )

    def search(
        self, index: RepositoryIndex | WorkspaceIndex, query: str, limit: int = 6
    ) -> list[tuple[str, int, int]] | None:
        with self._condition:
            if self._closed or self._ready != index.fingerprint:
                return None
            table = self._table
            lookup = self._lookup
        if table is None or not lookup:
            return []
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            with self._encoder_lock:
                if self._encoder is None:
                    self._encoder = SentenceTransformer(
                        str(self.model_path),
                        device="mps" if torch.backends.mps.is_available() else "cpu",
                        local_files_only=True,
                    )
                    self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        with self._encoder_lock:
            vector = self._encoder.encode_query(
                query[:3000], normalize_embeddings=True, show_progress_bar=False
            )
        matches = table.search(vector).limit(min(limit * 2, 32)).to_list()
        return [lookup[row["id"]] for row in matches if row["id"] in lookup][:limit]
