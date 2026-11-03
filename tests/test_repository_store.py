"""Persistent repository cache, FTS contents and scope revocation."""

import sqlite3
import stat
import threading

import pytest

from contextshift.repository import RepositoryIndex, scan_repository
from contextshift.repository_store import RepositoryStore


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build(root, store, selection=(".",), previous=None):
    return RepositoryIndex.build(
        root,
        scan_repository(root),
        selection,
        previous=previous,
        store=store,
    )


def test_restart_reuses_persistent_files_without_reading_contents(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    write(root, "src/cache.py", "def refresh_cache(): return old_value\n")
    store = RepositoryStore(tmp_path / "cache" / "repository.sqlite3")
    first = build(root, store)
    assert first.parsed_files == 1 and first.cache_hits == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged source content was read again")

    monkeypatch.setattr("contextshift.repository._read_source", forbidden)
    second = build(root, store)
    assert second.cache_hits == 1 and second.parsed_files == 0
    assert second.retrieve("refresh_cache").references[0].path == "src/cache.py"
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_changed_file_is_reparsed_and_fts_rows_are_replaced(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    source = write(root, "Service.java", "class Service { void oldCall() {} }\n")
    store = RepositoryStore(tmp_path / "repository.sqlite3")
    first = build(root, store)
    source.write_text("class Service { void newCall() {} }\n", encoding="utf-8")
    second = build(root, store, previous=first)
    assert second.parsed_files == 1
    with sqlite3.connect(store.path) as connection:
        old = connection.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'oldCall'"
        ).fetchone()[0]
        new = connection.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'newCall'"
        ).fetchone()[0]
    assert old == 0 and new > 0


def test_duplicate_java_symbol_on_one_line_is_stored_once(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    write(root, "Service.java", "class Service { Service() {} }\n")
    store = RepositoryStore(tmp_path / "repository.sqlite3")
    snapshot = build(root, store)
    assert snapshot.files["Service.java"].symbols.count((1, "Service")) == 2
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            """
            SELECT count(*) FROM symbols
            WHERE root = ? AND path = ? AND line = ? AND name = ?
            """,
            (str(root), "Service.java", 1, "Service"),
        ).fetchone()[0]
    assert rows == 1


def test_revoked_scope_is_removed_from_persistent_storage(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    write(root, "allowed.py", "def allowed(): return 1\n")
    write(root, "revoked.py", "def revoked_secret(): return 2\n")
    store = RepositoryStore(tmp_path / "repository.sqlite3")
    build(root, store)
    restricted = build(root, store, ("allowed.py",))
    assert set(restricted.files) == {"allowed.py"}
    with sqlite3.connect(store.path) as connection:
        paths = {
            row[0]
            for row in connection.execute(
                "SELECT path FROM files WHERE root = ?", (str(root),)
            )
        }
        leaked = connection.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'revoked_secret'"
        ).fetchone()[0]
    assert paths == {"allowed.py"} and leaked == 0


def test_cancelled_build_does_not_mutate_existing_cache(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    write(root, "module.py", "def stable(): return 1\n")
    store = RepositoryStore(tmp_path / "repository.sqlite3")
    build(root, store)
    cancelled = threading.Event()
    cancelled.set()
    try:
        RepositoryIndex.build(
            root,
            scan_repository(root),
            (),
            cancelled=cancelled,
            store=store,
        )
    except InterruptedError:
        pass
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM files").fetchone()[0] == 1


def test_cancelled_transaction_rolls_back_scope_revocation(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    write(root, "module.py", "def stable(): return 1\n")
    store = RepositoryStore(tmp_path / "repository.sqlite3")
    snapshot = build(root, store)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        store.sync(root, {}, cancelled)
    loaded = store.load(root, ("module.py",))
    assert loaded["module.py"].digest == snapshot.files["module.py"].digest
