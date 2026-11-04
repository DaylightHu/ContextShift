"""Exercise the multi-step code exploration against selected multi-root snapshots."""

from types import SimpleNamespace

import pytest

from contextshift.code_explorer import CodeExplorer
from contextshift.desktop.worker import AssistantWorker
from contextshift.llm.base import ModelRequest
from contextshift.repository import (
    ContextBundle,
    RepositoryIndex,
    SourceFile,
    SourceReference,
    WorkspaceIndex,
)


def _file(path, text):
    return SourceFile(path, text, (0, 0, 0), path, (), ())


def _call(name, arguments, call_id):
    import json

    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
    )


@pytest.mark.asyncio
async def test_agent_follows_names_across_selected_roots_and_verifies_lines(tmp_path):
    client_root, server_root = tmp_path / "app", tmp_path / "api"
    client = RepositoryIndex(
        client_root,
        (),
        "",
        {"Client.java": _file("Client.java", "api.vehicleInfo(id);")},
    )
    server = RepositoryIndex(
        server_root,
        (),
        "",
        {
            "VehicleController.java": _file(
                "VehicleController.java", "vehicleInfo(id) -> service.load(id);"
            )
        },
    )
    explorer = CodeExplorer(WorkspaceIndex((client, server)))
    steps = iter(
        [
            [_call("search_code", {"pattern": "vehicleInfo"}, "a")],
            [
                _call("read_file", {"path": "P1/Client.java"}, "b"),
                _call("read_file", {"path": "P2/VehicleController.java"}, "c"),
            ],
            [
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="调用入口和端点已确认")
                    ],
                )
            ],
        ]
    )
    requests = []

    async def create(**kwargs):
        requests.append({**kwargs, "input": list(kwargs["input"])})
        output = next(steps)
        return SimpleNamespace(output=output, output_text="调用入口和端点已确认")

    api = SimpleNamespace(responses=SimpleNamespace(create=create))
    bundle = await explorer.explore(api, model="test-model", query="入口如何联动端点？")
    assert len(requests) == 3
    assert "P1/Client.java:1" in requests[1]["input"][-1]["output"]
    assert {reference.workspace for reference in bundle.references} == {
        str(client_root),
        str(server_root),
    }
    assert "[S1]" in bundle.text and "[S2]" in bundle.text
    assert "service.load" in bundle.text
    assert explorer.execute("read_file", {"path": "P1/../../private.env"}) == (
        "Path is not among the selected files."
    )


@pytest.mark.asyncio
async def test_worker_uses_model_exploration_when_available(tmp_path):
    index = RepositoryIndex(
        tmp_path, (), "", {"a.py": _file("a.py", "def answer(): pass")}
    )
    worker = AssistantWorker(SimpleNamespace())
    worker._repository = index
    calls = []

    async def explore(scope, query, dialogue):
        calls.append((scope, query, dialogue))
        return ContextBundle(
            "Evidence from multiple tool calls",
            (SourceReference("a.py", 1, 1, "answer", "a.py", str(tmp_path)),),
            0.25,
        )

    worker._provider = SimpleNamespace(explore_code=explore)
    request = await worker._resolve_context(
        ModelRequest("input", "instruction", "question")
    )
    assert calls == [(index, "question", ())]
    assert "Evidence from multiple tool calls" in request.input
    assert len(worker._turn_sources) == 1
