"""A compact multi-root workspace; indexing never runs on the UI thread."""

import hashlib
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import (
    QSettings,
    QSignalBlocker,
    QStandardPaths,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from watchfiles import Change, watch

from contextshift.repository import (
    Inventory,
    RepositoryIndex,
    SourceReference,
    WorkspaceIndex,
    allowed_path,
    in_scope,
    scan_repository,
)
from contextshift.repository_store import RepositoryStore


@dataclass(slots=True)
class ProjectState:
    root: Path
    selection: tuple[str, ...] = ()
    notes: str = ""
    source_mode: str = "folder"
    selection_explicit: bool = False
    inventory: Inventory = field(default_factory=lambda: Inventory(()))


@dataclass(slots=True)
class ProjectUpdate:
    paths: set[str] = field(default_factory=set)
    rescan: bool = False


class WorkspaceWatcher(QThread):
    """Translate recursive native file-system notifications into Qt signals."""

    changed = Signal(object)
    failed = Signal(str)

    def __init__(self, roots: tuple[Path, ...], parent=None):
        super().__init__(parent)
        self.roots = roots
        self.cancelled = threading.Event()

    def run(self):
        try:
            for changes in watch(
                *(str(root) for root in self.roots),
                stop_event=self.cancelled,
                debounce=120,
                step=50,
                raise_interrupt=False,
            ):
                if self.cancelled.is_set():
                    return
                self.changed.emit(tuple(changes))
        except (OSError, RuntimeError) as exc:
            if not self.cancelled.is_set():
                self.failed.emit(str(exc))

    def stop(self):
        self.cancelled.set()


class IndexJob(QThread):
    result = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(
        self,
        revision,
        projects,
        previous,
        store,
        updates=None,
        parent=None,
    ):
        super().__init__(parent)
        self.revision = revision
        self.projects = projects
        self.previous = previous
        self.store = store
        self.updates = updates
        self.cancelled = threading.Event()

    def run(self):
        try:
            inventories = {}
            indexes = []
            for root, selection, notes, source_mode, current_inventory in self.projects:
                root_key = str(root)
                previous = self.previous.child(root) if self.previous else None
                update = (
                    self.updates.get(root_key) if self.updates is not None else None
                )
                if self.updates is not None and update is None and previous is not None:
                    inventories[root_key] = current_inventory
                    indexes.append(previous)
                    continue
                requested = selection if source_mode == "files" else ()
                if update is None or update.rescan:
                    inventory = scan_repository(root, self.cancelled, requested)
                else:
                    inventory = current_inventory
                changed_paths = None
                if update is not None and previous is not None:
                    changed = set(update.paths)
                    if update.rescan:
                        changed.update(
                            set(current_inventory.paths).symmetric_difference(
                                inventory.paths
                            )
                        )
                    changed_paths = frozenset(changed)
                index = RepositoryIndex.build(
                    root,
                    inventory,
                    selection,
                    notes,
                    previous=previous,
                    cancelled=self.cancelled,
                    store=self.store,
                    changed_paths=changed_paths,
                )
                inventories[root_key] = inventory
                indexes.append(index)
            if not self.cancelled.is_set():
                self.result.emit(
                    self.revision,
                    inventories,
                    WorkspaceIndex(tuple(indexes)),
                )
        except InterruptedError:
            pass
        except Exception as exc:  # noqa: BLE001 - report background file errors to Qt
            self.failed.emit(self.revision, str(exc))


class WorkspacePanel(QWidget):
    index_changed = Signal(object)
    focus_changed = Signal(str)
    idle = Signal()

    def __init__(
        self,
        preferences: QSettings,
        parent=None,
        *,
        store: RepositoryStore | None = None,
    ):
        super().__init__(parent)
        self.preferences = preferences
        cache_root = Path(
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.CacheLocation
            )
        )
        self.store = store or RepositoryStore(
            cache_root / "repository-index-v1-agentic.sqlite3"
        )
        self.projects: dict[str, ProjectState] = {}
        self.index: WorkspaceIndex | None = None
        self._previous_index: WorkspaceIndex | None = None
        self._active_root = ""
        self._revision = 0
        self._jobs: set[IndexJob] = set()
        self._watcher: WorkspaceWatcher | None = None
        self._watchers: set[WorkspaceWatcher] = set()
        self._pending_updates: dict[str, ProjectUpdate] = {}
        self._closing = False
        self._dialogs: list[QDialog] = []
        self.setMinimumWidth(180)
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 12, 8, 8)
        column.setSpacing(10)
        header = QHBoxLayout()
        self.title = QLabel("项目")
        self.title.setObjectName("workspaceTitle")
        header.addWidget(self.title, 1)
        self.open_button = QToolButton()
        self.open_button.setText("＋")
        self.open_button.setToolTip("追加多个文件或文件夹")
        self.open_button.setAccessibleName("追加项目资料")
        add_menu = QMenu(self.open_button)
        add_files = add_menu.addAction("追加文件…")
        add_files.triggered.connect(self.choose_files)
        add_folder = add_menu.addAction("追加文件夹…")
        add_folder.triggered.connect(self.choose_directory)
        self.open_button.setMenu(add_menu)
        self.open_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        header.addWidget(self.open_button)
        self.notes_button = QToolButton()
        self.notes_button.setText("✎")
        self.notes_button.setToolTip("当前项目说明：背景、职责、设计原因")
        self.notes_button.setAccessibleName("项目说明")
        self.notes_button.setEnabled(False)
        self.notes_button.clicked.connect(self.edit_notes)
        header.addWidget(self.notes_button)
        self.clear_button = QToolButton()
        self.clear_button.setText("×")
        self.clear_button.setToolTip("关闭全部项目，保留本场对话")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_project)
        header.addWidget(self.clear_button)
        column.addLayout(header)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setObjectName("projectTree")
        self.tree.setIndentation(15)
        self.tree.itemChanged.connect(self._selection_changed)
        self.tree.itemSelectionChanged.connect(self._focus_changed)
        self.tree.itemDoubleClicked.connect(self._open_item)
        column.addWidget(self.tree, 1)
        self.status = QLabel("追加文件夹后勾选范围；多选文件会直接加入。")
        self.status.setObjectName("workspaceStatus")
        self.status.setWordWrap(True)
        self.status.setToolTip(
            "所有追加项目会同时检索。排除密钥、二进制及构建目录；Git 仓库遵循忽略规则。"
        )
        column.addWidget(self.status)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._start_index)
        self._change_debounce = QTimer(self)
        self._change_debounce.setSingleShot(True)
        self._change_debounce.setInterval(160)
        self._change_debounce.timeout.connect(self._apply_file_changes)

    @property
    def root(self) -> Path | None:
        state = next(iter(self.projects.values()), None)
        return state.root if state else None

    @property
    def selection(self) -> tuple[str, ...]:
        state = self._active_state() or next(iter(self.projects.values()), None)
        return state.selection if state else ()

    @property
    def notes(self) -> str:
        state = self._active_state() or next(iter(self.projects.values()), None)
        return state.notes if state else ""

    @notes.setter
    def notes(self, value: str) -> None:
        state = self._active_state() or next(iter(self.projects.values()), None)
        if state is not None:
            state.notes = value[:6000]
            self._save_state(state)

    def restore(self):
        if self._closing or self.projects:
            return
        saved = self.preferences.value("workspace/roots", "")
        try:
            roots = json.loads(saved) if saved else []
        except (TypeError, ValueError):
            roots = []
        if not isinstance(roots, list):
            roots = []
        legacy = self.preferences.value("workspace/root", "")
        if not roots and legacy:
            roots = [legacy]
        for value in roots[:8]:
            path = Path(value)
            if path.is_dir():
                self._add_state(path.resolve())
        if self.projects:
            self._persist_roots()
            self._update_header()
            self.status.setText("正在读取工作区…")
            self._restart_watcher()
            self._start_index()

    @staticmethod
    def _project_key(root: Path, name: str) -> str:
        digest = hashlib.sha256(str(root).encode()).hexdigest()[:24]
        return f"workspace/projects/{digest}/{name}"

    def _load_state(
        self,
        root: Path,
        *,
        initial_selection: tuple[str, ...] | None = None,
        source_mode: str | None = None,
        selection_explicit: bool | None = None,
    ) -> ProjectState:
        mode = source_mode or str(
            self.preferences.value(self._project_key(root, "source_mode"), "folder")
        )
        if mode not in {"folder", "files"}:
            mode = "folder"
        explicit = (
            selection_explicit
            if selection_explicit is not None
            else self.preferences.value(
                self._project_key(root, "selection_explicit"), False, type=bool
            )
        )
        if initial_selection is None:
            try:
                saved = json.loads(
                    self.preferences.value(self._project_key(root, "selection"), "[]")
                )
                selection = (
                    tuple(value for value in saved if isinstance(value, str))
                    if isinstance(saved, list)
                    else ()
                )
            except (TypeError, ValueError):
                selection = ()
        else:
            selection = tuple(dict.fromkeys(initial_selection))
        if not explicit:
            selection = ()
        notes = str(self.preferences.value(self._project_key(root, "notes"), ""))[:6000]
        state = ProjectState(root, selection, notes, mode, bool(explicit))
        self._save_state(state)
        return state

    def _save_state(self, state: ProjectState) -> None:
        self.preferences.setValue(
            self._project_key(state.root, "selection"), json.dumps(state.selection)
        )
        self.preferences.setValue(
            self._project_key(state.root, "source_mode"), state.source_mode
        )
        self.preferences.setValue(
            self._project_key(state.root, "selection_explicit"),
            state.selection_explicit,
        )
        self.preferences.setValue(self._project_key(state.root, "notes"), state.notes)

    def _persist_roots(self) -> None:
        roots = [str(state.root) for state in self.projects.values()]
        self.preferences.setValue("workspace/roots", json.dumps(roots))
        if roots:
            self.preferences.setValue("workspace/root", roots[0])
        else:
            self.preferences.remove("workspace/root")

    def _add_state(self, root: Path, **options) -> ProjectState:
        root = root.resolve()
        key = str(root)
        if key in self.projects:
            self._active_root = key
            return self.projects[key]
        if len(self.projects) >= 8:
            raise ValueError("最多同时追加 8 个项目。")
        state = self._load_state(root, **options)
        self.projects[key] = state
        self._active_root = key
        return state

    def choose_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, "追加文件夹", str(self.root or Path.home())
        )
        if path:
            self.add_project(Path(path))

    def add_project(self, root: Path):
        try:
            root = root.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("请选择文件夹。")
            if str(root) in self.projects:
                self._active_root = str(root)
                self._update_header()
                self.status.setText("这个项目已经在工作区中")
                return
            self._add_state(
                root,
                initial_selection=(),
                source_mode="folder",
                selection_explicit=False,
            )
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self._workspace_changed("正在追加项目…")

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "追加多个文件",
            str(self.root or Path.home()),
            "文本和源代码 (*)",
        )
        if paths:
            self.add_files(tuple(Path(path).resolve() for path in paths))

    @staticmethod
    def _file_root(path: Path) -> Path:
        return next(
            (
                parent
                for parent in (path.parent, *path.parents)
                if (parent / ".git").exists()
            ),
            path.parent,
        )

    def add_files(self, paths: tuple[Path, ...]):
        files = tuple(dict.fromkeys(path for path in paths if path.is_file()))
        if not files:
            self.status.setText("没有可读取的文件")
            return
        groups: dict[Path, list[Path]] = {}
        for path in files:
            containing = next(
                (
                    state.root
                    for state in self.projects.values()
                    if path.is_relative_to(state.root)
                ),
                None,
            )
            root = containing or self._file_root(path)
            groups.setdefault(root, []).append(path)
        try:
            for root, group in groups.items():
                state = self.projects.get(str(root))
                if state is None:
                    selection = tuple(
                        path.relative_to(root).as_posix() for path in group
                    )
                    state = self._add_state(
                        root,
                        initial_selection=selection,
                        source_mode="files",
                        selection_explicit=True,
                    )
                else:
                    added = tuple(path.relative_to(root).as_posix() for path in group)
                    state.selection = tuple(dict.fromkeys((*state.selection, *added)))
                    state.selection_explicit = True
                    self._save_state(state)
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self._workspace_changed("正在追加文件…")

    def open_project(self, root: Path, **options):
        """Replace the workspace; retained for explicit project switching/tests."""
        self._close_previews()
        self._cancel_jobs()
        self.projects.clear()
        self.index = None
        self._previous_index = None
        state = self._add_state(root.resolve(), **options)
        self._active_root = str(state.root)
        self._workspace_changed("正在读取项目…")

    def clear_project(self):
        self._close_previews()
        self._cancel_jobs()
        self._debounce.stop()
        self._change_debounce.stop()
        self._pending_updates.clear()
        self._stop_watcher()
        self._revision += 1
        self.projects.clear()
        self.index = None
        self._previous_index = None
        self._active_root = ""
        self.preferences.remove("workspace/roots")
        self.preferences.remove("workspace/root")
        self.title.setText("项目")
        self.title.setToolTip("")
        self.tree.clear()
        self.notes_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.status.setText("追加文件夹后勾选范围；多选文件会直接加入。")
        self.index_changed.emit(None)
        self.focus_changed.emit("")

    def _workspace_changed(self, message: str):
        self._persist_roots()
        self._update_header()
        self._revision += 1
        self._close_previews()
        self._cancel_jobs()
        if self.index is not None:
            self._previous_index = self.index
        self.index = None
        self.tree.clear()
        self.index_changed.emit(None)
        self.focus_changed.emit("")
        self.status.setText(message)
        self._restart_watcher()
        self._start_index()

    def _active_state(self) -> ProjectState | None:
        return self.projects.get(self._active_root)

    def _update_header(self):
        count = len(self.projects)
        if count == 0:
            self.title.setText("项目")
            self.title.setToolTip("")
        elif count == 1:
            state = next(iter(self.projects.values()))
            self.title.setText(state.root.name)
            self.title.setToolTip(str(state.root))
        else:
            self.title.setText(f"工作区 · {count}")
            self.title.setToolTip("\n".join(self.projects))
        self.notes_button.setEnabled(bool(count))
        self.clear_button.setEnabled(bool(count))

    def _cancel_jobs(self):
        for job in self._jobs:
            job.cancelled.set()

    def _restart_watcher(self):
        self._stop_watcher()
        if self._closing or not self.projects:
            return
        watcher = WorkspaceWatcher(
            tuple(state.root for state in self.projects.values()), self
        )
        self._watcher = watcher
        self._watchers.add(watcher)
        watcher.changed.connect(self._file_changes)
        watcher.failed.connect(self._watch_failed)
        watcher.finished.connect(lambda: self._watcher_finished(watcher))
        watcher.start()

    def _stop_watcher(self):
        watcher, self._watcher = self._watcher, None
        if watcher is not None:
            watcher.stop()

    def _watcher_finished(self, watcher):
        self._watchers.discard(watcher)
        watcher.deleteLater()
        if self._closing and not self._jobs and not self._watchers:
            self.idle.emit()

    def _watch_failed(self, error):
        if self.sender() is not self._watcher or self._closing:
            return
        self.status.setText("文件监听失败")
        self.status.setToolTip(error)

    def _project_for_path(self, path: Path):
        for root_key, state in sorted(
            self.projects.items(), key=lambda item: len(item[0]), reverse=True
        ):
            try:
                return root_key, state, path.relative_to(state.root).as_posix()
            except ValueError:
                continue
        return None

    def _file_changes(self, changes):
        if self.sender() is not self._watcher or self._closing:
            return
        for change, value in changes:
            match = self._project_for_path(Path(value))
            if match is None:
                continue
            root_key, state, relative = match
            if relative == ".":
                safe = True
            else:
                safe = allowed_path(relative)
            if not safe:
                continue
            update = self._pending_updates.setdefault(root_key, ProjectUpdate())
            current = set(state.inventory.paths)
            is_directory = Path(value).is_dir() or any(
                path.startswith(relative.rstrip("/") + "/") for path in current
            )
            if (change in {Change.added, Change.deleted} or is_directory) and (
                state.source_mode == "folder" or in_scope(relative, state.selection)
            ):
                update.rescan = True
            affected = {
                path
                for path in current
                if path == relative or path.startswith(relative.rstrip("/") + "/")
            }
            if in_scope(relative, state.selection):
                affected.add(relative)
            update.paths.update(
                path for path in affected if in_scope(path, state.selection)
            )
            if not update.rescan and not update.paths:
                self._pending_updates.pop(root_key, None)
        if not self._pending_updates:
            return
        if self.index is None:
            self._cancel_jobs()
            self._revision += 1
            self.status.setText("工作区变化，正在重新读取…")
            self._debounce.start()
            self._pending_updates.clear()
            return
        self._change_debounce.start()

    def _apply_file_changes(self):
        if self._closing or not self._pending_updates:
            return
        updates, self._pending_updates = self._pending_updates, {}
        self.status.setText("已检测到文件变化 · 后台更新中")
        self._start_index(updates)

    def _start_index(self, updates=None):
        if self._closing or not self.projects:
            return
        self._cancel_jobs()
        self._revision += 1
        configs = tuple(
            (
                state.root,
                state.selection,
                state.notes,
                state.source_mode,
                state.inventory,
            )
            for state in self.projects.values()
        )
        job = IndexJob(
            self._revision,
            configs,
            self.index or self._previous_index,
            self.store,
            updates,
            self,
        )
        self._jobs.add(job)
        job.result.connect(self._indexed)
        job.failed.connect(self._failed)
        job.finished.connect(lambda: self._job_finished(job))
        job.start()

    def _job_finished(self, job):
        self._jobs.discard(job)
        job.deleteLater()
        if not self._jobs and (not self._closing or not self._watchers):
            self.idle.emit()

    def _indexed(self, revision, inventories, index):
        if self._closing or revision != self._revision:
            return
        previous = self.index
        changed = previous is None or previous.fingerprint != index.fingerprint
        if changed and previous is not None:
            self._close_previews()
        inventory_changed = False
        for key, state in self.projects.items():
            inventory = inventories.get(key, Inventory(()))
            if inventory.paths != state.inventory.paths:
                inventory_changed = True
            state.inventory = inventory
        self.index = index
        self._previous_index = index
        if inventory_changed or self.tree.topLevelItemCount() != len(self.projects):
            self._populate()
        total_inventory = sum(
            len(state.inventory.paths) for state in self.projects.values()
        )
        message = (
            f"{len(index.files)} / {total_inventory} 文件 · {len(self.projects)} 个项目"
        )
        parsed = sum(child.parsed_files for child in index.indexes)
        cached = sum(child.cache_hits for child in index.indexes)
        if cached and not parsed:
            message += " · 已复用本地索引"
        if index.skipped or any(
            state.inventory.truncated for state in self.projects.values()
        ):
            message += " · 部分未索引"
        self.status.setText(message)
        self.status.setToolTip(
            "\n".join(index.skipped[:20])
            + "\n所有已勾选项目会共同用于检索；文件修改后自动更新。"
        )
        if changed:
            self.index_changed.emit(index)

    def _failed(self, revision, error):
        if revision == self._revision and not self._closing:
            self.index = None
            self.status.setText("工作区读取失败")
            self.status.setToolTip(error)
            self.index_changed.emit(None)

    def _populate(self):
        current = self.tree.currentItem()
        focused = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        expanded = set()
        pending = [
            self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())
        ]
        while pending:
            item = pending.pop()
            if item.isExpanded():
                expanded.add(item.data(0, Qt.ItemDataRole.UserRole))
            pending.extend(item.child(i) for i in range(item.childCount()))
        with QSignalBlocker(self.tree):
            self.tree.clear()
            for root_key, state in self.projects.items():
                top_key = (root_key, ".")
                top = QTreeWidgetItem(self.tree, [state.root.name])
                top.setData(0, Qt.ItemDataRole.UserRole, top_key)
                top.setToolTip(0, str(state.root))
                top.setFlags(
                    top.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                )
                top.setCheckState(0, Qt.CheckState.Unchecked)
                parents = {".": top}
                for path in state.inventory.paths:
                    parts = path.split("/")
                    parent = top
                    for depth, part in enumerate(parts):
                        relative = "/".join(parts[: depth + 1])
                        is_directory = depth < len(parts) - 1
                        key = relative + "/" if is_directory else relative
                        if key not in parents:
                            item = QTreeWidgetItem(parent, [part])
                            item.setData(0, Qt.ItemDataRole.UserRole, (root_key, key))
                            item.setToolTip(0, f"{state.root.name}/{relative}")
                            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                            if is_directory:
                                flags |= Qt.ItemFlag.ItemIsAutoTristate
                            item.setFlags(flags)
                            item.setCheckState(0, Qt.CheckState.Unchecked)
                            parents[key] = item
                        parent = parents[key]
                    if any(
                        selected == "."
                        or path == selected.rstrip("/")
                        or path.startswith(selected.rstrip("/") + "/")
                        for selected in state.selection
                    ):
                        parent.setCheckState(0, Qt.CheckState.Checked)
                top.setExpanded(top_key in expanded or len(self.projects) <= 2)
                for key, item in parents.items():
                    item_key = (root_key, key)
                    if item_key in expanded:
                        item.setExpanded(True)
                    if item_key == focused:
                        self.tree.setCurrentItem(item)

    def _top_item(self, root_key: str) -> QTreeWidgetItem | None:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            value = item.data(0, Qt.ItemDataRole.UserRole)
            if value and value[0] == root_key:
                return item
        return None

    def _selected_paths(self, root_key: str) -> tuple[str, ...]:
        selected = []

        def visit(item):
            if item.checkState(0) == Qt.CheckState.Checked:
                selected.append(item.data(0, Qt.ItemDataRole.UserRole)[1])
            else:
                for child in range(item.childCount()):
                    visit(item.child(child))

        top = self._top_item(root_key)
        if top:
            visit(top)
        return tuple(selected)

    def _selection_changed(self, item, column):
        del column
        if self._closing:
            return
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if not value or value[0] not in self.projects:
            return
        root_key = value[0]
        state = self.projects[root_key]
        selected = self._selected_paths(root_key)
        if selected == state.selection:
            return
        state.selection = selected
        state.selection_explicit = True
        self._active_root = root_key
        self._save_state(state)
        self._invalidate()

    def _invalidate(self):
        self._revision += 1
        self._close_previews()
        self._cancel_jobs()
        if self.index is not None:
            self._previous_index = self.index
        self.index = None
        self.index_changed.emit(None)
        self.status.setText("正在更新索引…")
        self._debounce.start()

    def _focus_changed(self):
        item = self.tree.currentItem()
        value = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if not value:
            self.focus_changed.emit("")
            return
        root_key, path = value
        self._active_root = root_key
        self._update_header()
        key = WorkspaceIndex.file_key(root_key, path)
        self.focus_changed.emit(key if self.index and key in self.index.files else "")

    def _open_item(self, item, column):
        del column
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if not value:
            return
        root_key, path = value
        child = self.index.child(root_key) if self.index else None
        if child and path in child.files:
            file = child.files[path]
            self.open_source(
                SourceReference(path, 1, 1, "", file.digest, workspace=root_key)
            )
        elif path and path != "." and not path.endswith("/"):
            self.status.setText("勾选并索引后可预览文件")

    def edit_notes(self):
        state = self._active_state() or next(iter(self.projects.values()), None)
        if state is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"项目说明 · {state.root.name}")
        dialog.resize(600, 420)
        column = QVBoxLayout(dialog)
        editor = QPlainTextEdit(state.notes)
        editor.setPlaceholderText(
            "项目目标、你负责的部分、当时的约束、比较过的方案，以及最终选择的原因。\n这些资料单独保存，不会写入真实发言历史。"
        )
        column.addWidget(editor)
        counter = QLabel(f"{len(state.notes)} / 6000")
        column.addWidget(counter)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")

        def changed():
            count = len(editor.toPlainText())
            counter.setText(f"{count} / 6000")
            save.setEnabled(count <= 6000)

        editor.textChanged.connect(changed)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        column.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            state.notes = editor.toPlainText()
            self._save_state(state)
            self._invalidate()
        dialog.deleteLater()

    def open_source(self, reference: SourceReference):
        file = self.index.file_for(reference) if self.index else None
        if file is None:
            self.status.setText("来源已不在当前索引中")
            return
        root_name = Path(reference.workspace).name if reference.workspace else "项目"
        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"{root_name}/{reference.path}:{reference.start_line} · 索引快照"
        )
        dialog.resize(850, 600)
        self.focus_changed.emit(
            WorkspaceIndex.file_key(reference.workspace, reference.path)
        )
        column = QVBoxLayout(dialog)
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setPlainText(
            "\n".join(
                f"{number:4}  {line}"
                for number, line in enumerate(file.text.splitlines(), 1)
            )
        )
        cursor = QTextCursor(
            editor.document().findBlockByLineNumber(reference.start_line - 1)
        )
        editor.setTextCursor(cursor)
        selections = []
        for line in range(reference.start_line - 1, reference.end_line):
            block = editor.document().findBlockByLineNumber(line)
            if not block.isValid():
                break
            selection = QTextEdit.ExtraSelection()
            selection.cursor = QTextCursor(block)
            selection.format.setBackground(QColor("#293b43"))
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selections.append(selection)
        editor.setExtraSelections(selections)
        column.addWidget(editor)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._dialogs.append(dialog)
        dialog.destroyed.connect(
            lambda: self._dialogs.remove(dialog) if dialog in self._dialogs else None
        )
        dialog.show()
        editor.centerCursor()

    def open_source_in_vscode(self, reference: SourceReference) -> None:
        """Open a verified, selected source at its indexed line in VS Code."""

        if not self.index or self.index.file_for(reference) is None:
            self.status.setText("来源已不在当前索引中")
            return
        root_path = (
            Path(reference.workspace)
            if reference.workspace
            else self.index.indexes[0].root
        )
        root = root_path.resolve()
        target = (root / reference.path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            self.status.setText("来源文件已移动或不可访问")
            return

        code = shutil.which("code") or shutil.which("code-insiders")
        if code is None and sys.platform == "darwin":
            for applications in (Path("/Applications"), Path.home() / "Applications"):
                bundled = (
                    applications
                    / "Visual Studio Code.app/Contents/Resources/app/bin/code"
                )
                if bundled.is_file():
                    code = str(bundled)
                    break
        if code is None:
            self.status.setText("未找到 VS Code，请安装 code 命令")
            return
        try:
            subprocess.Popen(
                [code, "--goto", f"{target}:{max(1, reference.start_line)}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self.status.setText("无法启动 VS Code")

    def shutdown(self):
        self._closing = True
        self._debounce.stop()
        self._change_debounce.stop()
        self._pending_updates.clear()
        self._cancel_jobs()
        for watcher in tuple(self._watchers):
            watcher.stop()
        self._watcher = None
        self._close_previews()
        return not self._jobs and not self._watchers

    def _close_previews(self):
        for dialog in tuple(self._dialogs):
            dialog.close()
