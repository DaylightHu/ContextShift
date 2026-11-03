"""Qt scope selection, project persistence, refresh and background shutdown."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from contextshift.desktop import workspace as module
from contextshift.desktop.workspace import WorkspacePanel


def wait(app, predicate):
    deadline = time.monotonic() + 4
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


@pytest.fixture
def panel(tmp_path):
    app = QApplication.instance() or QApplication([])
    prefs = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    view = WorkspacePanel(prefs)
    yield app, view
    view.shutdown()
    wait(app, lambda: not view._jobs)
    view.close()


def item(panel, path):
    pending = [panel.tree.topLevelItem(0)]
    while pending:
        current = pending.pop()
        if current.data(0, Qt.ItemDataRole.UserRole) == path:
            return current
        pending.extend(current.child(i) for i in range(current.childCount()))
    raise AssertionError(path)


def test_folder_selection_is_partial_and_restores_per_project(panel, tmp_path):
    app, view = panel
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src/cache.py").write_text("def refresh_cache(): pass\n")
    (root / "src/backend.py").write_text("def load_data(): pass\n")
    (root / "notes.md").write_text("project notes")
    view.open_project(root)
    wait(app, lambda: view.index is not None)
    assert not view.index.files  # Opening a directory does not authorize content.
    item(view, "src/").setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and len(view.index.files) == 2)
    assert item(view, ".").checkState(0) == Qt.CheckState.PartiallyChecked
    item(view, "src/backend.py").setCheckState(0, Qt.CheckState.Unchecked)
    assert view.index is None  # The old scope is revoked synchronously.
    wait(app, lambda: view.index is not None)
    assert set(view.index.files) == {"src/cache.py"}
    wait(app, lambda: not view._jobs)
    view.open_project(root)
    wait(app, lambda: view.index is not None)
    assert set(view.index.files) == {"src/cache.py"}


def test_selected_folder_includes_new_files_after_background_refresh(panel, tmp_path):
    app, view = panel
    root = tmp_path / "project"
    root.mkdir()
    (root / "one.py").write_text("def one(): pass\n")
    view.open_project(root)
    wait(app, lambda: view.index is not None)
    item(view, ".").setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.files))
    assert view.selection == (".",)
    (root / "two.py").write_text("def two(): pass\n")
    wait(app, lambda: not view._jobs)
    view._poll()
    wait(app, lambda: "two.py" in view.index.files)
    assert item(view, "two.py").checkState(0) == Qt.CheckState.Checked
    assert view.selection == (".",)


def test_close_cancels_index_and_leaves_no_background_qthread(
    panel, tmp_path, monkeypatch
):
    app, view = panel
    entered = threading.Event()

    def scan(root, cancelled):
        entered.set()
        assert cancelled.wait(3)
        raise InterruptedError

    monkeypatch.setattr(module, "scan_repository", scan)
    view.open_project(tmp_path)
    wait(app, entered.is_set)
    assert not view.shutdown()
    wait(app, lambda: not view._jobs)
    assert not view._refresh.isActive()


def test_old_project_results_and_identical_file_references_cannot_leak(panel, tmp_path):
    app, view = panel
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "module.py").write_text("def logic(): return 1\n")
    view.open_project(a)
    wait(app, lambda: view.index is not None)
    item(view, ".").setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.files))
    reference = view.index.retrieve("logic").references[0]
    old_revision, old_inventory, old_index = view._revision, view._inventory, view.index
    view.open_project(b)
    wait(app, lambda: view.index is not None)
    item(view, ".").setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.files))
    assert view.index.files["module.py"].digest == reference.digest
    view.open_source(reference)
    assert not view._dialogs
    view._indexed(old_revision, old_inventory, old_index)
    assert view.index.root == b
