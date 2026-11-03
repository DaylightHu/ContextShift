"""Qt scope selection, project persistence, refresh and background shutdown."""

import hashlib
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from contextshift.desktop import workspace as module
from contextshift.desktop.workspace import WorkspacePanel
from contextshift.repository_store import RepositoryStore


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
    view = WorkspacePanel(
        prefs, store=RepositoryStore(tmp_path / "repository-index.sqlite3")
    )
    yield app, view
    view.shutdown()
    wait(app, lambda: not view._jobs)
    view.close()


def item(panel, path, root=None):
    pending = [
        panel.tree.topLevelItem(index)
        for index in range(panel.tree.topLevelItemCount())
    ]
    while pending:
        current = pending.pop()
        value = current.data(0, Qt.ItemDataRole.UserRole)
        if value and value[1] == path and (root is None or value[0] == str(root)):
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
    assert set(view.index.child(root).files) == {"src/cache.py"}
    wait(app, lambda: not view._jobs)
    view.open_project(root)
    wait(app, lambda: view.index is not None)
    assert set(view.index.child(root).files) == {"src/cache.py"}


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
    wait(app, lambda: "two.py" in view.index.child(root).files)
    assert item(view, "two.py").checkState(0) == Qt.CheckState.Checked
    assert view.selection == (".",)


def test_close_cancels_index_and_leaves_no_background_qthread(
    panel, tmp_path, monkeypatch
):
    app, view = panel
    entered = threading.Event()

    def scan(root, cancelled, selection):
        assert selection == ()
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
    old_revision, old_index = view._revision, view.index
    old_inventories = {key: state.inventory for key, state in view.projects.items()}
    view.open_project(b)
    wait(app, lambda: view.index is not None)
    item(view, ".").setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.files))
    assert view.index.child(b).files["module.py"].digest == reference.digest
    view.open_source(reference)
    assert not view._dialogs
    view._indexed(old_revision, old_inventories, old_index)
    assert view.index.root == b


def test_add_files_authorizes_multiple_files_without_clicking_each_checkbox(
    panel, tmp_path
):
    app, view = panel
    root = tmp_path / "project"
    root.mkdir()
    first = root / "Controller.java"
    second = root / "Service.java"
    ignored = root / "Other.java"
    first.write_text("class Controller {}\n")
    second.write_text("class Service {}\n")
    ignored.write_text("class Other {}\n")
    view.open_project(root)
    wait(app, lambda: view.index is not None)
    view.add_files((first, second))
    wait(app, lambda: view.index is not None and len(view.index.files) == 2)
    assert set(view.index.child(root).files) == {"Controller.java", "Service.java"}
    assert item(view, "Controller.java").checkState(0) == Qt.CheckState.Checked
    assert item(view, "Service.java").checkState(0) == Qt.CheckState.Checked
    assert item(view, "Other.java").checkState(0) == Qt.CheckState.Unchecked


def test_choosing_a_folder_does_not_authorize_files(panel, tmp_path, monkeypatch):
    app, view = panel
    root = tmp_path / "project"
    root.mkdir()
    (root / "Main.java").write_text("class Main {}\n")
    monkeypatch.setattr(
        module.QFileDialog, "getExistingDirectory", lambda *args: str(root)
    )
    view.choose_directory()
    wait(app, lambda: view.index is not None)
    assert view.selection == ()
    assert not view.index.files
    assert item(view, "Main.java").checkState(0) == Qt.CheckState.Unchecked


def test_implicit_selection_from_previous_version_is_not_restored(panel, tmp_path):
    app, view = panel
    root = (tmp_path / "project").resolve()
    (root / "app").mkdir(parents=True)
    (root / "app/Main.java").write_text("class Main {}\n")
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    view.preferences.setValue("workspace/root", str(root))
    view.preferences.setValue(f"workspace/projects/{digest}/selection", '["app/"]')
    view.restore()
    wait(app, lambda: view.index is not None)
    assert view.selection == ()
    assert not view.index.files
    assert item(view, "app/").checkState(0) == Qt.CheckState.Unchecked


def test_appending_a_folder_keeps_both_projects_and_searches_them(panel, tmp_path):
    app, view = panel
    client = tmp_path / "nio_packages"
    platform = tmp_path / "nasa-platform"
    client.mkdir()
    platform.mkdir()
    (client / "RouteActivity.java").write_text(
        "class RouteActivity { void upload() { PlatformClient.send(); } }\n"
    )
    (platform / "PlatformClient.java").write_text(
        "class PlatformClient { static void send() { dispatch(); } }\n"
    )
    view.open_project(client)
    wait(app, lambda: view.index is not None)
    item(view, ".", client).setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.child(client).files))
    view.add_project(platform)
    wait(app, lambda: view.index is not None and len(view.projects) == 2)
    assert view.index.child(client).files
    assert not view.index.child(platform).files
    assert view.tree.topLevelItemCount() == 2
    item(view, ".", platform).setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.child(platform).files))
    bundle = view.index.retrieve("PlatformClient.send 在哪里实现？")
    assert {Path(reference.workspace).name for reference in bundle.references} == {
        "nio_packages",
        "nasa-platform",
    }


def test_all_appended_roots_restore_together(panel, tmp_path):
    app, view = panel
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "one.py").write_text("def one(): pass\n")
    (second / "two.py").write_text("def two(): pass\n")
    view.open_project(first)
    wait(app, lambda: view.index is not None)
    item(view, ".", first).setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.child(first).files))
    view.add_project(second)
    wait(app, lambda: view.index is not None and len(view.projects) == 2)
    item(view, ".", second).setCheckState(0, Qt.CheckState.Checked)
    wait(app, lambda: view.index is not None and bool(view.index.child(second).files))
    view.preferences.sync()
    restored = WorkspacePanel(view.preferences)
    restored.restore()
    wait(app, lambda: restored.index is not None)
    assert set(restored.projects) == {str(first), str(second)}
    assert restored.index.child(first).files
    assert restored.index.child(second).files
    restored.shutdown()
    wait(app, lambda: not restored._jobs)
    restored.close()
