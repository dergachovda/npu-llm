from npu_llm.__main__ import available_devices


def test_cpu_is_always_available() -> None:
    assert "CPU" in available_devices()