"""Warmup is real work; opened streams and warmed models are independent gates."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from contextshift.speech import AudioConfig, listen
from contextshift.speech.asr import SenseVoiceTranscriber
from contextshift.speech.audio import MicrophoneStream
from contextshift.speech.vad import FsmnVad


def test_warmup_runs_inference_and_clears_synthetic_vad_state(tmp_path):
    np = pytest.importorskip("numpy")
    samples = []

    class Model:
        def generate(self, **kwargs):
            samples.append(kwargs["input"])
            if "fs" in kwargs:
                kwargs["cache"]["synthetic"] = True
                return [{"value": [[0, -1]]}]
            return [{"text": "discard this warmup result"}]

    asr = SenseVoiceTranscriber(tmp_path)
    asr._model = Model()
    vad = FsmnVad(tmp_path)
    vad._model = Model()
    asr.warm_up()
    vad.warm_up()
    assert len(samples) == 2
    assert samples[0].shape == (16_000,) and samples[0].dtype == np.float32
    assert samples[1].shape == (3_200,) and not samples[1].any()
    assert vad.pending_start_ms is None and vad._cache == {}


def test_both_channels_are_prepared_independently_and_cancellation_is_checked(
    monkeypatch,
):
    calls = []

    class Vad:
        def __init__(self, *args, **kwargs):
            pass

        def warm_up(self):
            calls.append("vad")

    class Asr:
        def __init__(self, *args, **kwargs):
            pass

        def warm_up(self):
            calls.append("asr")

    monkeypatch.setattr(listen, "FsmnVad", Vad)
    monkeypatch.setattr(listen, "SenseVoiceTranscriber", Asr)
    options = {
        "asr_model_dir": "unused",
        "vad_model_dir": "unused",
        "microphone_config": AudioConfig(device=1),
        "system_config": AudioConfig(device=0),
    }
    prepared = listen.prepare_channels(**options)
    assert calls == ["vad", "asr", "vad", "asr"]
    assert prepared["microphone"].vad is not prepared["system"].vad
    assert prepared["microphone"].transcriber is not prepared["system"].transcriber
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        listen.prepare_channels(**options, cancelled=cancelled)
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_two_open_streams_are_required_for_ready_and_ready_fires_once(
    monkeypatch,
):
    callbacks = {}
    entered = asyncio.Event()
    announced = []

    async def listener(*, source, on_ready, **kwargs):
        callbacks[source] = on_ready
        if len(callbacks) == 2:
            entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(listen, "run_audio_listener", listener)
    task = asyncio.create_task(
        listen.run_dual_audio_listener(
            asr_model_dir="unused",
            vad_model_dir="unused",
            microphone_config=AudioConfig(device=1),
            system_config=AudioConfig(device=0),
            on_ready=lambda: announced.append(True),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        callbacks["microphone"]()
        callbacks["microphone"]()
        assert announced == []
        callbacks["system"]()
        callbacks["system"]()
        assert announced == [True]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_listener_reuses_prepared_models_instead_of_lazy_loading_again(
    monkeypatch,
):
    ready = asyncio.Event()
    closed = []

    class Stream:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def next_block(self):
            raise TimeoutError

    def forbidden(*args, **kwargs):
        raise AssertionError("model loaded twice")

    monkeypatch.setattr(listen, "MicrophoneStream", Stream)
    monkeypatch.setattr(listen, "FsmnVad", forbidden)
    monkeypatch.setattr(listen, "SenseVoiceTranscriber", forbidden)
    config = AudioConfig(device=1)
    prepared = listen.PreparedChannel(config, SimpleNamespace(), SimpleNamespace())
    task = asyncio.create_task(
        listen.run_audio_listener(
            source="microphone",
            asr_model_dir="unused",
            vad_model_dir="unused",
            audio_config=config,
            prepared=prepared,
            on_ready=ready.set,
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert closed == [True]


@pytest.mark.parametrize("phase", ["start", "stop"])
def test_audio_stream_is_closed_even_if_start_or_stop_fails(monkeypatch, phase):
    from contextshift.speech import audio

    calls = []

    class Stream:
        def start(self):
            calls.append("start")
            if phase == "start":
                raise RuntimeError("start failed")

        def stop(self):
            calls.append("stop")
            if phase == "stop":
                raise RuntimeError("stop failed")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(
        audio,
        "_sounddevice",
        lambda: SimpleNamespace(InputStream=lambda **kwargs: Stream()),
    )
    stream = MicrophoneStream(AudioConfig())
    with pytest.raises(RuntimeError), stream:
        pass
    assert calls[-1] == "close" and stream._stream is None
