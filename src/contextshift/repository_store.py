"""Crash-safe persistent storage for local repository indexes."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path

from contextshift.repository import Chunk, SourceFile, SourceReference

_SCHEMA_VERSION = 1


class RepositoryStore:
    """Persist parsed source files while keeping one connection per operation."""

    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.path, timeout=10)
        self.path.chmod(0o600)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                root TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                inode INTEGER NOT NULL,
                digest TEXT NOT NULL,
                text TEXT NOT NULL,
                imports_json TEXT NOT NULL,
                PRIMARY KEY (root, path)
            );
            CREATE TABLE IF NOT EXISTS symbols (
                root TEXT NOT NULL,
                path TEXT NOT NULL,
                line INTEGER NOT NULL,
                name TEXT NOT NULL,
                PRIMARY KEY (root, path, line, name),
                FOREIGN KEY (root, path) REFERENCES files(root, path)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS chunks (
                root TEXT NOT NULL,
                path TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                text TEXT NOT NULL,
                tokens_json TEXT NOT NULL,
                PRIMARY KEY (root, path, ordinal),
                FOREIGN KEY (root, path) REFERENCES files(root, path)
                    ON DELETE CASCADE
            );
            """
        )
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    root UNINDEXED,
                    path,
                    symbol,
                    text,
                    tokenize='trigram'
                )
                """
            )
        except sqlite3.OperationalError as exc:
            connection.close()
            raise RuntimeError("当前 Python 的 SQLite 未启用 FTS5 trigram。") from exc
        version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if version is not None and int(version[0]) != _SCHEMA_VERSION:
            connection.close()
            raise RuntimeError("代码索引版本不兼容，请清理本地索引后重试。")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        connection.commit()
        return connection

    def load(self, root: Path, paths: tuple[str, ...]) -> dict[str, SourceFile]:
        """Load only currently authorized paths; stale scopes never enter memory."""
        if not paths or not self.path.exists():
            return {}
        root_value = str(root)
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TEMP TABLE requested(path TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT OR IGNORE INTO requested(path) VALUES(?)",
                ((path,) for path in paths),
            )
            file_rows = connection.execute(
                """
                SELECT f.path, f.mtime_ns, f.size, f.inode, f.digest, f.text,
                       f.imports_json
                FROM files AS f
                JOIN requested AS r ON r.path = f.path
                WHERE f.root = ?
                """,
                (root_value,),
            ).fetchall()
            symbols: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for path, line, name in connection.execute(
                """
                SELECT s.path, s.line, s.name
                FROM symbols AS s
                JOIN requested AS r ON r.path = s.path
                WHERE s.root = ?
                ORDER BY s.path, s.line, s.name
                """,
                (root_value,),
            ):
                symbols[path].append((line, name))
            chunk_rows: dict[str, list[tuple]] = defaultdict(list)
            for row in connection.execute(
                """
                SELECT c.path, c.ordinal, c.start_line, c.end_line, c.symbol,
                       c.text, c.tokens_json
                FROM chunks AS c
                JOIN requested AS r ON r.path = c.path
                WHERE c.root = ?
                ORDER BY c.path, c.ordinal
                """,
                (root_value,),
            ):
                chunk_rows[row[0]].append(row[1:])
            loaded = {}
            for path, mtime, size, inode, digest, text, imports_json in file_rows:
                references = []
                for _, start, end, symbol, chunk_text, tokens_json in chunk_rows[path]:
                    references.append(
                        Chunk(
                            SourceReference(path, start, end, symbol, digest),
                            chunk_text,
                            json.loads(tokens_json),
                        )
                    )
                loaded[path] = SourceFile(
                    path,
                    text,
                    (mtime, size, inode),
                    digest,
                    tuple(symbols[path]),
                    tuple(references),
                    tuple(json.loads(imports_json)),
                )
            return loaded
        except (json.JSONDecodeError, sqlite3.DatabaseError, TypeError, ValueError):
            return {}
        finally:
            connection.close()

    def sync(
        self,
        root: Path,
        files: dict[str, SourceFile],
        cancelled: threading.Event | None = None,
        *,
        changed_paths: frozenset[str] | None = None,
    ) -> None:
        """Atomically add changed files and purge content outside the live scope."""
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("项目索引已取消。")
        root_value = str(root)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("项目索引已取消。")
            existing = dict(
                connection.execute(
                    "SELECT path, digest FROM files WHERE root = ?", (root_value,)
                )
            )
            removed = set(existing) - set(files)
            for path in removed:
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError("项目索引已取消。")
                connection.execute(
                    "DELETE FROM chunks_fts WHERE root = ? AND path = ?",
                    (root_value, path),
                )
                connection.execute(
                    "DELETE FROM files WHERE root = ? AND path = ?",
                    (root_value, path),
                )
            updates = (
                files.items()
                if changed_paths is None
                else ((path, files[path]) for path in changed_paths if path in files)
            )
            for path, source in updates:
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError("项目索引已取消。")
                unchanged_content = existing.get(path) == source.digest
                connection.execute(
                    """
                    INSERT INTO files(
                        root, path, mtime_ns, size, inode, digest, text, imports_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(root, path) DO UPDATE SET
                        mtime_ns=excluded.mtime_ns,
                        size=excluded.size,
                        inode=excluded.inode,
                        digest=excluded.digest,
                        text=excluded.text,
                        imports_json=excluded.imports_json
                    """,
                    (
                        root_value,
                        path,
                        *source.signature,
                        source.digest,
                        source.text,
                        json.dumps(source.imports, ensure_ascii=False),
                    ),
                )
                if unchanged_content:
                    continue
                connection.execute(
                    "DELETE FROM symbols WHERE root = ? AND path = ?",
                    (root_value, path),
                )
                connection.execute(
                    "DELETE FROM chunks WHERE root = ? AND path = ?",
                    (root_value, path),
                )
                connection.execute(
                    "DELETE FROM chunks_fts WHERE root = ? AND path = ?",
                    (root_value, path),
                )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO symbols(root, path, line, name)
                    VALUES(?, ?, ?, ?)
                    """,
                    ((root_value, path, line, name) for line, name in source.symbols),
                )
                for ordinal, chunk in enumerate(source.chunks):
                    reference = chunk.reference
                    connection.execute(
                        """
                        INSERT INTO chunks(
                            root, path, ordinal, start_line, end_line, symbol,
                            text, tokens_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            root_value,
                            path,
                            ordinal,
                            reference.start_line,
                            reference.end_line,
                            reference.symbol,
                            chunk.text,
                            json.dumps(dict(chunk.tokens), ensure_ascii=False),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO chunks_fts(root, path, symbol, text)
                        VALUES(?, ?, ?, ?)
                        """,
                        (root_value, path, reference.symbol, chunk.text),
                    )
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("项目索引已取消。")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
