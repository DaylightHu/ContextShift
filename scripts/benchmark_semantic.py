"""Measure local semantic indexing/search on private code without sending it out."""

import argparse
from pathlib import Path
from time import perf_counter

from contextshift.repository import RepositoryIndex, WorkspaceIndex, scan_repository
from contextshift.semantic_index import SemanticIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="A typical question about the selected code")
    parser.add_argument("projects", type=Path, nargs="+", help="Local project roots")
    args = parser.parse_args()
    semantic = SemanticIndex()
    if not semantic.available:
        parser.error(f"Model not found at {semantic.model_path}")
    started = perf_counter()
    indexes = []
    for root in args.projects:
        root = root.expanduser().absolute()
        inventory = scan_repository(root)
        indexes.append(RepositoryIndex.build(root, inventory, (".",)))
    index = WorkspaceIndex(tuple(indexes))
    scan_seconds = perf_counter() - started
    semantic._build(index, 0)
    build_seconds = perf_counter() - started - scan_seconds
    query_started = perf_counter()
    matches = semantic.search(index, args.question)
    query_seconds = perf_counter() - query_started
    print(f"files={sum(len(project.files) for project in indexes)}")
    print(
        f"chunks={sum(len(source.chunks) for project in indexes for source in project.files.values())}"
    )
    print(f"scan_seconds={scan_seconds:.3f}")
    print(f"index_seconds={build_seconds:.3f}")
    print(f"query_seconds={query_seconds:.3f}")
    print(f"candidates={len(matches or [])}")
    semantic.close()


if __name__ == "__main__":
    main()
