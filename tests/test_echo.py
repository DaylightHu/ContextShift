"""Real native AEC, alignment, reference loss and capture error propagation."""

import threading
from types import SimpleNamespace

import pytest

from contextshift.speech.audio import AudioConfig, AudioPacket, MicrophoneStream
from contextshift.speech.echo import EchoProcessor
from contextshift.speech.errors import SpeechConfigurationError

np = pytest.importorskip("numpy")


def processor(**kwargs):
    pytest.importorskip("pywebrtc_audio")
    return EchoProcessor(16_000, **kwargs)


def test_native_aec_suppresses_delayed_multipath_echo_and_keeps_double_talk():
    rng = np.random.default_rng(93)
    rate, count, block = 16_000, 16_000 * 12, 3200
    far = (rng.standard_normal(count) * 0.1).astype(np.float32)
    echo = np.zeros(count, dtype=np.float32)
    echo[1280:] += 0.6 * far[:-1280]
    echo[1800:] += 0.2 * far[:-1800]
    t = np.arange(count) / rate
    # Independent harmonic, amplitude-modulated near-end signal, not a recording.
    near = (
        (
            np.sin(2 * np.pi * 173 * t)
            + 0.5 * np.sin(2 * np.pi * 346 * t)
            + 0.3 * np.sin(2 * np.pi * 519 * t)
        )
        * 0.08
        * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    ).astype(np.float32)
    near[: rate * 7] = 0
    mixed = echo + near
    aec = processor()
    output = []
    for i in range(0, count, block):
        aec.publish(AudioPacket(far[i : i + block], i / rate))
        output.append(aec.clean(AudioPacket(mixed[i : i + block], i / rate)))
    output = np.concatenate(output)
    far_only = slice(rate * 4, rate * 7)
    erle = 10 * np.log10(np.mean(echo[far_only] ** 2) / np.mean(output[far_only] ** 2))
    assert erle > 15, f"insufficient echo reduction: {erle:.1f} dB"
    # AEC's internal frame/filter delay must not be mistaken for lost speech.
    expected = near[rate * 9 : -1024]
    correlations = [
        np.corrcoef(expected, output[rate * 9 + lag : count - 1024 + lag])[0, 1]
        for lag in range(1024)
    ]
    lag = int(np.argmax(correlations))
    actual = output[rate * 9 + lag : count - 1024 + lag]
    gain = np.dot(actual, expected) / np.dot(expected, expected)
    assert correlations[lag] > 0.75
    assert gain > 0.4  # Not replaced by silence while the far end is active.


def test_silence_is_a_valid_reference_but_missing_reference_is_an_error():
    aec = processor(reference_timeout=0.01)
    silence = np.zeros(3200, dtype=np.float32)
    aec.publish(AudioPacket(silence, 10.0))
    assert aec.clean(AudioPacket(silence, 10.0)).shape == (3200,)
    with pytest.raises(SpeechConfigurationError, match="参考中断"):
        aec.clean(AudioPacket(silence, 10.2))


def test_reference_is_aligned_by_adc_time_not_queue_order():
    aec = processor()
    calls = []
    aec._processor = SimpleNamespace(
        process=lambda near, far: calls.append(far) or near
    )
    aec.publish(AudioPacket(np.arange(3200, dtype=np.float32), 10.0))
    aec.publish(AudioPacket(np.arange(3200, 6400, dtype=np.float32), 10.2))
    aec.clean(AudioPacket(np.zeros(3200, dtype=np.float32), 10.05))
    np.testing.assert_allclose(calls[0], np.arange(800, 4000), atol=0.01)


def test_initial_overlap_is_skipped_before_ready_but_later_gap_is_fatal():
    aec = processor()
    silence = np.zeros(3200, dtype=np.float32)
    aec.publish(AudioPacket(silence, 10.1))
    assert aec.clean(AudioPacket(silence, 10.0)) is None
    aec.publish(AudioPacket(silence, 10.3))
    assert aec.clean(AudioPacket(silence, 10.2)) is not None
    with pytest.raises(SpeechConfigurationError, match="麦克风音频不连续"):
        aec.clean(AudioPacket(silence, 10.6))
    with pytest.raises(SpeechConfigurationError, match="参考流不连续"):
        aec.publish(AudioPacket(silence, 10.8))


def test_close_wakes_waiter_and_reference_buffer_is_bounded():
    aec = processor(reference_timeout=5)
    silence = np.zeros(3200, dtype=np.float32)
    errors = []
    entered = threading.Event()

    def wait():
        entered.set()
        try:
            aec.clean(AudioPacket(silence, 0.0))
        except SpeechConfigurationError as exc:
            errors.append(exc)

    thread = threading.Thread(target=wait)
    thread.start()
    assert entered.wait(1)
    aec.close()
    thread.join(1)
    assert not thread.is_alive() and errors
    aec = processor()
    for i in range(100):
        aec.publish(AudioPacket(silence, i * 0.2))
    assert len(aec._packets) <= 26


@pytest.mark.parametrize("failure", ["overflow", "queue", "reference"])
def test_strict_capture_never_hides_discontinuities(monkeypatch, failure):
    from contextshift.speech import audio

    callbacks = []
    closed = []

    def make_stream(**kwargs):
        callbacks.append(kwargs["callback"])
        return SimpleNamespace(
            start=lambda: None, stop=lambda: None, close=lambda: closed.append(True)
        )

    def broken_reference(packet):
        raise SpeechConfigurationError("reference failure")

    monkeypatch.setattr(
        audio, "_sounddevice", lambda: SimpleNamespace(InputStream=make_stream)
    )
    with MicrophoneStream(
        AudioConfig(),
        strict=True,
        on_packet=broken_reference if failure == "reference" else None,
    ) as stream:
        for i in range(17 if failure == "queue" else 1):
            callbacks[0](
                np.zeros((3200, 1), dtype=np.float32),
                3200,
                SimpleNamespace(
                    currentTime=10 + i * 0.2, inputBufferAdcTime=9.8 + i * 0.2
                ),
                "input overflow" if failure == "overflow" else False,
            )
        with pytest.raises(SpeechConfigurationError):
            stream.next_packet()
    assert closed == [True]
