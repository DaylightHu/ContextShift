"""Scope, source fidelity, follow-up retrieval and local refresh behavior."""

import re
import subprocess
import threading

import pytest

from contextshift.repository import (
    Inventory,
    RepositoryIndex,
    WorkspaceIndex,
    allowed_path,
    scan_repository,
)


def write(root, path, text):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file


def index(root, selection=(".",), notes="", previous=None):
    return RepositoryIndex.build(
        root, scan_repository(root), selection, notes, previous=previous
    )


def test_only_checked_files_are_read_and_secret_binary_and_symlinks_are_excluded(
    tmp_path,
):
    write(tmp_path, "src/cache.py", "def refresh_cache():\n    return 'SELECTED'\n")
    write(tmp_path, "private/unselected.py", "UNSELECTED_SECRET = 123\n")
    write(tmp_path, ".env", "API_KEY=PRIVATE_KEY\n")
    write(tmp_path, ".npmrc", "TOKEN=PRIVATE_TOKEN\n")
    write(tmp_path, "cert.pem", "PRIVATE_CERT")
    write(tmp_path, "node_modules/package/index.js", "DEPENDENCY_SECRET")
    write(tmp_path, "data.unknown", "BINARY\0DATA")
    (tmp_path / "shortcut.py").symlink_to(tmp_path / "private/unselected.py")
    inventory = scan_repository(tmp_path)
    assert "src/cache.py" in inventory.paths
    assert not any(
        p in inventory.paths
        for p in [
            ".env",
            ".npmrc",
            "cert.pem",
            "shortcut.py",
            "node_modules/package/index.js",
        ]
    )
    snapshot = index(tmp_path, ("src/",))
    assert set(snapshot.files) == {"src/cache.py"}
    context = snapshot.retrieve("缓存在哪里").text
    assert "SELECTED" in context
    assert "UNSELECTED_SECRET" not in context and "PRIVATE_KEY" not in context
    snapshot = index(tmp_path)
    assert "data.unknown" not in snapshot.files
    assert snapshot.skipped


def test_git_ignore_is_respected_and_nested_project_paths_are_relative(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "ignored/\n*.generated.py\n")
    write(tmp_path, "ignored/secret.py", "SECRET")
    write(tmp_path, "src/autogen.generated.py", "GENERATED")
    write(tmp_path, "src/main.py", "def main(): pass")
    write(tmp_path, "other.py", "OTHER")
    assert set(scan_repository(tmp_path / "src").paths) == {"main.py"}
    assert "ignored/secret.py" not in scan_repository(tmp_path).paths
    assert "src/autogen.generated.py" not in scan_repository(tmp_path).paths


@pytest.mark.parametrize(
    "value",
    [
        "../outside.py",
        "/tmp/outside.py",
        ".",
        "src/../../outside.py",
        "src/.env.local",
        "x/.git/config",
    ],
)
def test_malformed_and_sensitive_file_paths_are_rejected(value):
    assert not allowed_path(value)


def test_parent_symlink_swap_after_scan_cannot_escape_root(tmp_path):
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    write(root, "src/impl.py", "SAFE")
    write(outside, "impl.py", "OUTSIDE_SECRET")
    inventory = scan_repository(root)
    (root / "src/impl.py").unlink()
    (root / "src").rmdir()
    (root / "src").symlink_to(outside, target_is_directory=True)
    snapshot = RepositoryIndex.build(root, inventory, (".",))
    assert not snapshot.files and snapshot.skipped
    assert "OUTSIDE_SECRET" not in snapshot.retrieve("impl").text


def test_chinese_followups_keep_cache_focus_and_a_new_topic_can_move_to_auth(tmp_path):
    write(
        tmp_path,
        "src/cache.py",
        """class Cache:
    def refresh_cache(self):
        try:
            return self.load()
        except Exception:
            return self.previous_cache

    def refresh_concurrently(self):
        with self.refresh_lock:
            return self.refresh_cache()
""",
    )
    write(
        tmp_path,
        "src/login.py",
        "def authenticate_login(token):\n    return verify_token(token)\n",
    )
    snapshot = index(tmp_path)
    first = snapshot.retrieve("缓存刷新在哪里？")
    assert first.references[0].path == "src/cache.py"
    second = snapshot.retrieve(
        "失败了怎么办？", dialogue=("缓存刷新在哪里？",), focus=first.references
    )
    assert second.references[0].path == "src/cache.py"
    assert "previous_cache" in second.text
    third = snapshot.retrieve(
        "那两个线程同时刷新呢？",
        dialogue=("缓存刷新在哪里？", "失败了怎么办？"),
        focus=second.references,
    )
    assert "refresh_lock" in third.text
    fourth = snapshot.retrieve(
        "登录鉴权入口在哪里？", dialogue=("缓存刷新在哪里？",), focus=third.references
    )
    assert fourth.references[0].path == "src/login.py"


def test_references_and_preview_lines_are_derived_from_real_source(tmp_path):
    source = "\n".join(f"line_{i} = {i}" for i in range(1, 160))
    write(tmp_path, "module.py", source)
    snapshot = index(tmp_path)
    bundle = snapshot.retrieve("line_140")
    assert bundle.references
    for ref in bundle.references:
        assert ref.workspace == str(tmp_path)
        assert ref.digest == snapshot.files[ref.path].digest
        assert ref.start_line >= 1 and ref.end_line <= 159
        assert (
            f"{ref.start_line}: line_{ref.start_line} = {ref.start_line}" in bundle.text
        )


def test_overview_notes_and_strict_context_budget(tmp_path):
    write(
        tmp_path, "pyproject.toml", '[project]\ndependencies = ["PySide6", "openai"]\n'
    )
    write(tmp_path, "src/component.py", "def process_audio():\n    return 1\n")
    snapshot = index(tmp_path, notes="由我负责界面。为了低延迟选择本地处理。")
    bundle = snapshot.retrieve("项目用了什么技术？")
    assert bundle.references[0].path == "pyproject.toml"
    assert "为了低延迟" in bundle.text and "PySide6" in bundle.text
    snapshot = index(tmp_path, notes="\n\t" * 4000)
    for budget in [512, 1024, 20000]:
        assert len(snapshot.retrieve("技术", max_characters=budget).text) <= budget


def test_no_match_is_explicit_and_index_refresh_revokes_removed_files(tmp_path):
    file = write(tmp_path, "module.py", "def old_logic(): return 1\n")
    before = index(tmp_path)
    assert (
        "没有找到直接相关代码" in before.retrieve("totally_unrelated_searchterm").text
    )
    unchanged = index(tmp_path, previous=before)
    assert unchanged.files["module.py"] is before.files["module.py"]
    file.write_text("def new_logic(): return 2\n")
    after = index(tmp_path, previous=before)
    assert before.fingerprint != after.fingerprint
    assert "old_logic" not in after.retrieve("new_logic").text
    revoked = index(tmp_path, selection=(), previous=after)
    assert not revoked.files and "new_logic" not in revoked.retrieve("new_logic").text


def test_cancelled_index_does_no_source_work(tmp_path):
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        RepositoryIndex.build(
            tmp_path, Inventory(("module.py",)), (".",), cancelled=event
        )
    snapshot = RepositoryIndex(tmp_path, (), "", {})
    with pytest.raises(InterruptedError):
        snapshot.retrieve("cancelled question", cancelled=event)


def test_followup_can_expand_to_called_code_in_another_selected_module(tmp_path):
    write(
        tmp_path,
        "cache.py",
        "from backend import load_data\n\ndef refresh_cache():\n    return load_data()\n",
    )
    write(
        tmp_path,
        "backend.py",
        "def load_data():\n    try:\n        return read_storage()\n    except Exception:\n        return previous_value\n",
    )
    for i in range(8):
        write(
            tmp_path,
            f"unrelated_{i}.py",
            f"def error_handler_{i}():\n    raise Exception('error failure fallback')\n",
        )
    snapshot = index(tmp_path)
    first = snapshot.retrieve("refresh_cache 在哪里？")
    focus = tuple(r for r in first.references if r.path == "cache.py")
    assert focus
    followup = snapshot.retrieve("它失败了呢？", focus=focus)
    assert "backend.py" in {r.path for r in followup.references}
    assert "previous_value" in followup.text


def test_python_definitions_do_not_share_chunks_with_unrelated_vocabulary(tmp_path):
    write(
        tmp_path,
        "source.py",
        'WORDS = ["cache", "refresh", "concurrency", "thread", "lock"]\n\ndef refresh_cache():\n    return load_data()\n',
    )
    snapshot = index(tmp_path)
    chunks = snapshot.files["source.py"].chunks
    implementation = next(
        c for c in chunks if c.reference.symbol.startswith("refresh_cache")
    )
    assert "WORDS" not in implementation.text
    assert implementation.reference.start_line == 3
    assert implementation.reference.end_line == 4


def test_project_overview_does_not_get_stuck_on_previous_code_focus(tmp_path):
    write(tmp_path, "cache.py", "def refresh_cache(): return old_value\n")
    write(
        tmp_path, "pyproject.toml", '[project]\ndependencies = ["PySide6", "openai"]\n'
    )
    snapshot = index(tmp_path)
    previous = snapshot.retrieve("缓存刷新在哪里？")
    bundle = snapshot.retrieve("这个项目用了什么技术栈？", focus=previous.references)
    assert bundle.references[0].path == "pyproject.toml"


def test_technology_overview_can_use_imports_from_a_selected_source_folder(tmp_path):
    write(
        tmp_path,
        "src/app.py",
        "from PySide6.QtWidgets import QApplication\nimport numpy as np\n\ndef main():\n    return QApplication([])\n",
    )
    snapshot = index(tmp_path, ("src/",))
    bundle = snapshot.retrieve("这个项目用了什么技术栈？")
    assert "PySide6.QtWidgets" in bundle.text and "numpy" in bundle.text
    assert bundle.references[0].path == "src/app.py"


def test_search_can_locate_contextshift_preemption_and_stale_output_guards():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    names = (
        "src/contextshift/live.py",
        "src/contextshift/session.py",
        "src/contextshift/runtime/turn_controller.py",
        "src/contextshift/speech/listen.py",
        "src/contextshift/cli.py",
        "src/contextshift/repository.py",
        "src/contextshift/desktop/worker.py",
        "tests/test_live.py",
    )
    snapshot = RepositoryIndex.build(root, Inventory(names), (".",))
    first = snapshot.retrieve("系统音频开始讲话时在哪里取消旧回答？")
    assert "src/contextshift/live.py" in {r.path for r in first.references}
    focus = tuple(r for r in first.references if r.path == "src/contextshift/live.py")
    second = snapshot.retrieve("那旧的 token 为什么不会显示？", focus=focus)
    assert "src/contextshift/session.py" in {r.path for r in second.references}
    assert "src/contextshift/runtime/turn_controller.py" in {
        r.path for r in second.references
    }


def test_java_retrieval_promotes_interface_caller_and_network_factory(tmp_path):
    write(
        tmp_path,
        "TaskUploadApiService.java",
        """package com.nasa.api;
import retrofit2.http.POST;
public interface TaskUploadApiService {
    @POST("task/upload")
    @NasaRequestResponse
    LiveData<ApiResponse<Task>> uploadTask(Task body);
}
""",
    )
    write(
        tmp_path,
        "NasaTaskRepository.java",
        """public class NasaTaskRepository {
    private final TaskUploadApiService service =
        TspNetManager.getServer("nasa-app-api").create(TaskUploadApiService.class);
    public LiveData<ApiResponse<Task>> upload(Task task) {
        return service.uploadTask(task);
    }
}
""",
    )
    write(
        tmp_path,
        "TspNetManager.java",
        """public final class TspNetManager {
    public static Retrofit getServer(String name) { return servers.get(name); }
}
""",
    )
    write(
        tmp_path,
        "AudioRecorderTools.java",
        """public class AudioRecorderTools {
    public void record() { logger.info("app api recorder"); }
}
""",
    )
    write(tmp_path, "build.gradle", 'implementation project(":app")\n')
    snapshot = index(tmp_path)
    bundle = snapshot.retrieve("你目前的架构是怎么跟 nasa-app-api 联动的？")
    core = {reference.path for reference in bundle.references[:3]}
    assert "NasaTaskRepository.java" in core
    assert "TaskUploadApiService.java" in core
    assert "AudioRecorderTools.java" not in core
    assert any(
        name == "uploadTask"
        for _, name in snapshot.files["TaskUploadApiService.java"].symbols
    )
    assert "TspNetManager.getServer" in bundle.text


def test_module_literal_and_endpoint_outrank_interceptor_and_dto_across_roots(
    tmp_path,
):
    platform = tmp_path / "nasa-platform"
    client = tmp_path / "nio_packages_apps_nasa"
    write(
        platform,
        "nasa-app-api/src/VehicleInfoController.java",
        """@RestController
@RequestMapping("/vehicle")
public class VehicleInfoController {
    @GetMapping("/info")
    VehicleInfoResponse vehicleInfo() {
        return vehicleInfoService.queryVehicleInfo();
    }
}
""",
    )
    write(
        platform,
        "nasa-app-api/src/VehicleInfoService.java",
        """public class VehicleInfoService {
    VehicleInfoResponse queryVehicleInfo() { return vehicleGateway.load(); }
}
""",
    )
    write(
        platform,
        "nasa-web-api/src/VehicleInfoRequest.java",
        "public class VehicleInfoRequest { String vehicleId; }\n",
    )
    write(
        platform,
        "common/src/OkHttpRetryInterceptor.java",
        "class OkHttpRetryInterceptor { Response doRequest(Request request) {} }\n",
    )
    write(
        client,
        "src/NasaAppVehicleRepository.java",
        """class NasaAppVehicleRepository {
    private final VehicleApi api = TspNetManager.getServer("nasa-app-api")
        .create(VehicleApi.class);
    VehicleInfo loadVehicleInfo() { return api.getVehicleInfo(); }
}
""",
    )
    write(
        client,
        "src/VehicleApi.java",
        """interface VehicleApi {
    @GET("vehicle/info") VehicleInfo getVehicleInfo();
}
""",
    )
    workspace = WorkspaceIndex((index(platform), index(client)))

    bundle = workspace.retrieve(
        "ok能不能帮我讲一下nasa-app是怎么调用nasa-app-api的数据接口查到车辆信息的"
    )
    paths = [reference.path for reference in bundle.references]
    assert paths[0] == "nasa-app-api/src/VehicleInfoController.java"
    assert "nasa-app-api/src/VehicleInfoController.java" in paths[:3]
    assert "nasa-web-api/src/VehicleInfoRequest.java" not in paths[:3]
    assert "common/src/OkHttpRetryInterceptor.java" not in paths[:3]
    assert "src/NasaAppVehicleRepository.java" in paths
    assert "src/VehicleApi.java" in paths


def test_project_instructions_are_direct_speech_not_coaching():
    from contextshift.repository import PROJECT_INSTRUCTIONS

    assert "第一句直接回答问题" in PROJECT_INSTRUCTIONS
    assert "不要写‘可答’" in PROJECT_INSTRUCTIONS


def test_workspace_retrieval_combines_roots_and_renumbers_sources(tmp_path):
    client = tmp_path / "nio_packages"
    platform = tmp_path / "nasa-platform"
    write(
        client,
        "RouteActivity.java",
        "class RouteActivity { void upload() { PlatformClient.send(); } }\n",
    )
    write(
        platform,
        "PlatformClient.java",
        "class PlatformClient { static void send() { dispatch(); } }\n",
    )
    workspace = WorkspaceIndex((index(client), index(platform)))
    bundle = workspace.retrieve("PlatformClient.send 在哪里实现？")
    assert {reference.workspace for reference in bundle.references} == {
        str(client),
        str(platform),
    }
    citations = re.findall(r"\[S(\d+)\]", bundle.text)
    assert citations == [str(number) for number in range(1, len(citations) + 1)]
    assert len(bundle.text) <= 20000
