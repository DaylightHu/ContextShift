"""Verify selected-snapshot isolation and incremental vector reuse."""

import sys
from types import SimpleNamespace

import pytest

from contextshift.repository import (
    Chunk,
    RepositoryIndex,
    SourceFile,
    SourceReference,
)
from contextshift.semantic_index import SemanticIndex


def _index(root, path, content):
    reference = SourceReference(path, 1, 1, "", content, workspace=str(root))
    file = SourceFile(
        path, content, (0, 0, 0), content, (), (Chunk(reference, content, {}),)
    )
    return RepositoryIndex(root, (path,), "", {path: file})


def test_semantic_index_reuses_vectors_and_drops_deselected_files(
    tmp_path, monkeypatch
):
    databases = {}
    encoded = []
    progress = []

    class Vector:
        def __init__(self, number):
            self.number = number

        def tolist(self):
            return [self.number, 1.0]

    class Encoder:
        def __init__(self, path, *, device, local_files_only):
            assert path == str(tmp_path / "model")
            assert local_files_only
            assert device == "mps"

        def encode_document(self, texts, **kwargs):
            encoded.extend(texts)
            return [Vector(float("alpha" in text)) for text in texts]

        def encode_query(self, query, **kwargs):
            return [float("alpha" in query), 1.0]

    class Table:
        def __init__(self, rows):
            self.rows = list(rows)

        def add(self, rows):
            self.rows.extend(rows)

        def delete(self, expression):
            ids = set(
                expression.split("(", 1)[1].rstrip(")").replace("'", "").split(", ")
            )
            self.rows = [row for row in self.rows if row["id"] not in ids]

        def search(self, vector=None):
            if vector is None:
                return SimpleNamespace(
                    select=lambda _: SimpleNamespace(
                        to_arrow=lambda: SimpleNamespace(
                            column=lambda _: SimpleNamespace(
                                to_pylist=lambda: [row["id"] for row in self.rows]
                            )
                        )
                    )
                )
            rows = sorted(self.rows, key=lambda row: abs(row["vector"][0] - vector[0]))
            return SimpleNamespace(
                limit=lambda count: SimpleNamespace(to_list=lambda: rows[:count])
            )

    class DB:
        def __init__(self):
            self.table = None

        def list_tables(self):
            return SimpleNamespace(tables=["chunks"] if self.table else [])

        def open_table(self, name):
            return self.table

        def create_table(self, name, data):
            self.table = Table(data)
            return self.table

    monkeypatch.setitem(
        sys.modules,
        "lancedb",
        SimpleNamespace(connect=lambda path: databases.setdefault(path, DB())),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )

    service = SemanticIndex(tmp_path / "model", tmp_path / "cache", progress.append)
    original = _index(tmp_path / "project", "A.java", "alpha method")
    service._build(original, 0)
    assert service.search(original, "alpha") == [("P1/A.java", 1, 1)]
    assert len(encoded) == 1
    assert progress[-1].stage == "ready"
    assert progress[-1].device == "mps"
    assert (progress[-1].completed, progress[-1].total) == (1, 1)

    service._build(original, 0)
    assert len(encoded) == 1
    assert progress[-1].reused == 1

    replacement = _index(tmp_path / "project", "B.java", "beta method")
    service._generation = 1
    service._ready = None
    assert service.search(replacement, "alpha") is None
    service._build(replacement, 1)
    assert len(encoded) == 2
    assert progress[-1].stage == "ready"
    assert progress[-1].reused == 0
    assert service.search(replacement, "alpha") == [("P1/B.java", 1, 1)]
    assert len(next(iter(databases.values())).table.rows) == 1
    service.close()
    assert service.search(replacement, "alpha") is None


def test_lancedb_persists_and_reuses_selected_vectors(tmp_path, monkeypatch):
    pytest.importorskip("lancedb")
    encoded = []

    class Vector:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_document(self, texts, **kwargs):
            encoded.extend(texts)
            return [Vector([1.0 if "alpha" in text else 0.0, 1.0]) for text in texts]

        def encode_query(self, query, **kwargs):
            return [1.0 if "alpha" in query else 0.0, 1.0]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )

    cache = tmp_path / "cache"
    root = tmp_path / "project"
    first = _index(root, "A.java", "alpha method")
    service = SemanticIndex(tmp_path / "model", cache)
    service._build(first, 0)
    assert service.search(first, "alpha") == [("P1/A.java", 1, 1)]

    restarted = SemanticIndex(tmp_path / "model", cache)
    restarted._build(first, 0)
    assert len(encoded) == 1
    assert restarted.search(first, "alpha") == [("P1/A.java", 1, 1)]

    second = _index(root, "B.java", "beta method")
    restarted._generation = 1
    restarted._build(second, 1)
    assert restarted.search(second, "alpha") == [("P1/B.java", 1, 1)]
    assert len(encoded) == 2
