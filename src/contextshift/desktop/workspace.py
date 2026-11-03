"""A compact project tree; indexing and refresh never run on the UI thread."""

import hashlib
import json
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from contextshift.repository import (
    Inventory,
    RepositoryIndex,
    SourceReference,
    scan_repository,
)


class IndexJob(QThread):
    result = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(self, revision, root, selection, notes, previous, parent=None):
        super().__init__(parent)
        self.revision, self.root, self.selection = revision, root, selection
        self.notes, self.previous = notes, previous
        self.cancelled = threading.Event()

    def run(self):
        try:
            inventory = scan_repository(self.root, self.cancelled)
            index = RepositoryIndex.build(
                self.root,
                inventory,
                self.selection,
                self.notes,
                previous=self.previous,
                cancelled=self.cancelled,
            )
            if not self.cancelled.is_set():
                self.result.emit(self.revision, inventory, index)
        except InterruptedError:
            pass
        except Exception as exc:  # noqa: BLE001 - report background file errors to Qt
            self.failed.emit(self.revision, str(exc))


class WorkspacePanel(QWidget):
    index_changed = Signal(object)
    focus_changed = Signal(str)
    idle = Signal()

    def __init__(self, preferences: QSettings, parent=None):
        super().__init__(parent)
        self.preferences = preferences
        self.root: Path | None = None
        self.index: RepositoryIndex | None = None
        self.selection: tuple[str, ...] = ()
        self.notes = ""
        self._inventory = Inventory(())
        self._revision = 0
        self._jobs: set[IndexJob] = set()
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
        self.open_button.setToolTip("打开项目文件夹")
        self.open_button.setAccessibleName("打开项目文件夹")
        self.open_button.clicked.connect(self.choose_directory)
        header.addWidget(self.open_button)
        self.notes_button = QToolButton()
        self.notes_button.setText("✎")
        self.notes_button.setToolTip("项目说明：背景、职责、设计原因")
        self.notes_button.setAccessibleName("项目说明")
        self.notes_button.setEnabled(False)
        self.notes_button.clicked.connect(self.edit_notes)
        header.addWidget(self.notes_button)
        self.clear_button = QToolButton()
        self.clear_button.setText("×")
        self.clear_button.setToolTip("关闭项目，保留本场对话")
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
        self.status = QLabel("打开项目后，勾选可用于回答的文件。")
        self.status.setObjectName("workspaceStatus")
        self.status.setWordWrap(True)
        self.status.setToolTip(
            "只读取勾选范围。排除密钥、二进制及构建目录；Git 仓库遵循忽略规则。"
        )
        column.addWidget(self.status)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._start_index)
        self._refresh = QTimer(self)
        self._refresh.setInterval(3000)
        self._refresh.timeout.connect(self._poll)

    def restore(self):
        if self._closing or self.root is not None:
            return
        saved = self.preferences.value("workspace/root", "")
        if saved and Path(saved).is_dir():
            self.open_project(Path(saved))

    def _key(self, name):
        digest = hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        return f"workspace/projects/{digest}/{name}"

    def choose_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, "打开项目", str(self.root or Path.home())
        )
        if path:
            self.open_project(Path(path))

    def open_project(self, root: Path):
        self._close_previews()
        self._cancel_jobs()
        self.root = root.resolve()
        self.index = None
        self._inventory = Inventory(())
        self._revision += 1
        try:
            saved = json.loads(self.preferences.value(self._key("selection"), "[]"))
            self.selection = (
                tuple(p for p in saved if isinstance(p, str))
                if isinstance(saved, list)
                else ()
            )
        except (ValueError, TypeError):
            self.selection = ()
        self.notes = str(self.preferences.value(self._key("notes"), ""))[:6000]
        self.preferences.setValue("workspace/root", str(self.root))
        self.title.setText(self.root.name)
        self.title.setToolTip(str(self.root))
        self.notes_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.tree.clear()
        self.index_changed.emit(None)
        self.focus_changed.emit("")
        self.status.setText("正在读取项目…")
        self._start_index()
        self._refresh.start()

    def clear_project(self):
        self._close_previews()
        self._cancel_jobs()
        self._debounce.stop()
        self._refresh.stop()
        self._revision += 1
        self.root = self.index = None
        self.selection, self.notes = (), ""
        self._inventory = Inventory(())
        self.preferences.remove("workspace/root")
        self.title.setText("项目")
        self.tree.clear()
        self.notes_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.status.setText("打开项目后，勾选可用于回答的文件。")
        self.index_changed.emit(None)
        self.focus_changed.emit("")

    def _cancel_jobs(self):
        for job in self._jobs:
            job.cancelled.set()

    def _start_index(self):
        if self._closing or self.root is None:
            return
        self._cancel_jobs()
        self._revision += 1
        job = IndexJob(
            self._revision, self.root, self.selection, self.notes, self.index, self
        )
        self._jobs.add(job)
        job.result.connect(self._indexed)
        job.failed.connect(self._failed)
        job.finished.connect(lambda: self._job_finished(job))
        job.start()

    def _job_finished(self, job):
        self._jobs.discard(job)
        job.deleteLater()
        if not self._jobs:
            self.idle.emit()

    def _indexed(self, revision, inventory, index):
        if self._closing or revision != self._revision:
            return
        previous = self.index
        changed = previous is None or previous.fingerprint != index.fingerprint
        if changed and previous is not None:
            self._close_previews()
        if inventory.paths != self._inventory.paths:
            self._populate(inventory)
        self._inventory, self.index = inventory, index
        message = f"{len(index.files)} / {len(inventory.paths)} 文件"
        if index.skipped or inventory.truncated:
            message += " · 部分未索引"
        self.status.setText(message)
        self.status.setToolTip(
            "\n".join(index.skipped[:20])
            + ("\n文件列表超过 5000 项，请缩小范围。" if inventory.truncated else "")
            + "\n仅所选代码可用于回答；文件修改后自动更新。"
        )
        if changed:
            self.index_changed.emit(index)

    def _failed(self, revision, error):
        if revision == self._revision and not self._closing:
            self.index = None
            self.status.setText("项目读取失败")
            self.status.setToolTip(error)
            self.index_changed.emit(None)

    def _populate(self, inventory):
        focused = (
            self.tree.currentItem().data(0, Qt.ItemDataRole.UserRole)
            if self.tree.currentItem()
            else None
        )
        expanded = set()
        iterator = []
        for i in range(self.tree.topLevelItemCount()):
            iterator.append(self.tree.topLevelItem(i))
        while iterator:
            item = iterator.pop()
            if item.isExpanded():
                expanded.add(item.data(0, Qt.ItemDataRole.UserRole))
            iterator.extend(item.child(i) for i in range(item.childCount()))
        with QSignalBlocker(self.tree):
            self.tree.clear()
            top = QTreeWidgetItem(self.tree, [self.root.name])
            top.setData(0, Qt.ItemDataRole.UserRole, ".")
            top.setFlags(
                top.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            top.setCheckState(0, Qt.CheckState.Unchecked)
            parents = {".": top}
            for path in inventory.paths:
                parts = path.split("/")
                parent = top
                for depth, part in enumerate(parts):
                    relative = "/".join(parts[: depth + 1])
                    is_directory = depth < len(parts) - 1
                    key = relative + "/" if is_directory else relative
                    if key not in parents:
                        item = QTreeWidgetItem(parent, [part])
                        item.setData(0, Qt.ItemDataRole.UserRole, key)
                        item.setToolTip(0, relative)
                        flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                        if is_directory:
                            flags |= Qt.ItemFlag.ItemIsAutoTristate
                        item.setFlags(flags)
                        item.setCheckState(0, Qt.CheckState.Unchecked)
                        parents[key] = item
                    parent = parents[key]
                if any(
                    s == "."
                    or path == s.rstrip("/")
                    or path.startswith(s.rstrip("/") + "/")
                    for s in self.selection
                ):
                    parent.setCheckState(0, Qt.CheckState.Checked)
            top.setExpanded(True)
            for key, item in parents.items():
                if key in expanded:
                    item.setExpanded(True)
                if key == focused:
                    self.tree.setCurrentItem(item)

    def _selected_paths(self):
        selected = []

        def visit(item):
            if item.checkState(0) == Qt.CheckState.Checked:
                selected.append(item.data(0, Qt.ItemDataRole.UserRole))
            else:
                for i in range(item.childCount()):
                    visit(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))
        return tuple(selected)

    def _selection_changed(self, *args):
        if self.root is None or self._closing:
            return
        selected = self._selected_paths()
        if selected == self.selection:
            return
        self.selection = selected
        self.preferences.setValue(self._key("selection"), json.dumps(selected))
        self._invalidate()

    def _invalidate(self):
        # Revoke old content immediately, before the background rebuild starts.
        self._revision += 1
        self._close_previews()
        self._cancel_jobs()
        self.index = None
        self.index_changed.emit(None)
        self.status.setText("正在更新索引…")
        self._debounce.start()

    def _focus_changed(self):
        item = self.tree.currentItem()
        path = item.data(0, Qt.ItemDataRole.UserRole) if item else ""
        self.focus_changed.emit(path if self.index and path in self.index.files else "")

    def _open_item(self, item, column):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if self.index and path in self.index.files:
            file = self.index.files[path]
            self.open_source(SourceReference(path, 1, 1, "", file.digest))
        elif path and path != "." and not path.endswith("/"):
            self.status.setText("勾选并索引后可预览文件")

    def _poll(self):
        if not self._jobs and not self._debounce.isActive():
            self._start_index()

    def edit_notes(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("项目说明")
        dialog.resize(600, 420)
        column = QVBoxLayout(dialog)
        editor = QPlainTextEdit(self.notes)
        editor.setPlaceholderText(
            "项目目标、你负责的部分、当时的约束、比较过的方案，以及最终选择的原因。\n这些资料单独保存，不会写入真实发言历史。"
        )
        column.addWidget(editor)
        counter = QLabel(f"{len(self.notes)} / 6000")
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
            self.notes = editor.toPlainText()
            self.preferences.setValue(self._key("notes"), self.notes)
            self._invalidate()
        dialog.deleteLater()

    def open_source(self, reference: SourceReference):
        index = self.index
        file = index.files.get(reference.path) if index else None
        if (
            file is None
            or file.digest != reference.digest
            or (reference.workspace and reference.workspace != str(index.root))
        ):
            self.status.setText("来源已不在当前索引中")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{reference.path}:{reference.start_line} · 索引快照")
        dialog.resize(850, 600)
        self.focus_changed.emit(reference.path)
        column = QVBoxLayout(dialog)
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setPlainText(
            "\n".join(
                f"{i:4}  {line}" for i, line in enumerate(file.text.splitlines(), 1)
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

    def shutdown(self):
        self._closing = True
        self._debounce.stop()
        self._refresh.stop()
        self._cancel_jobs()
        self._close_previews()
        return not self._jobs

    def _close_previews(self):
        for dialog in tuple(self._dialogs):
            dialog.close()
