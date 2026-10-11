"""Перевірка NPU, запуск LLM та вимірювання продуктивності."""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import openvino as ov
import openvino_genai as ov_genai
import psutil

MODEL_DIR = Path("models/qwen_npu_sym")

PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat is an NPU? Answer in two sentences.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Pipeline(Protocol):
    """Опис LLM-пайплайна, який використовує програма."""

    def generate(
        self,
        inputs: str,
        max_new_tokens: int,
        streamer: Callable[[str], None],
    ) -> str:
        """Генерує текст за промптом і передає фрагменти в streamer."""
        ...


@dataclass
class StreamStats:
    """Збирає час і кількість фрагментів під час генерації."""

    start: float
    first_token_at: float | None = None
    chunks: int = 0

    def on_chunk(self, _text: str) -> None:
        """Викликається на кожен новий фрагмент тексту.

        Args:
            _text: Фрагмент тексту (не використовується, потрібна лише кількість).
        """
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self.chunks += 1


def available_devices() -> list[str]:
    """Повертає пристрої, які бачить OpenVINO.

    Returns:
        Назви пристроїв, наприклад ['CPU', 'GPU', 'NPU'].
    """
    return list(ov.Core().available_devices)


def build_pipeline(model_dir: Path, device: str) -> Pipeline:
    """Створює LLM-пайплайн на вказаному пристрої.

    Args:
        model_dir: Каталог з експортованою моделлю OpenVINO.
        device: Назва пристрою, наприклад 'NPU' або 'CPU'.

    Returns:
        Пайплайн з типізованим інтерфейсом.
    """
    # openvino_genai не має анотацій типів, тому приводимо об'єкт до нашого протоколу.
    return cast(Pipeline, ov_genai.LLMPipeline(str(model_dir), device))


def rss_mb() -> float:
    """Повертає RAM, яку використовує поточний процес, у мегабайтах.

    Returns:
        Розмір резидентної пам'яті в МБ.
    """
    return psutil.Process().memory_info().rss / 1024**2


def main() -> None:
    """Запускає генерацію та виводить метрики."""
    devices = available_devices()
    print(f"Доступні пристрої: {devices}")

    if not MODEL_DIR.exists():
        sys.exit(f"Модель не знайдено: {MODEL_DIR}")

    device = "NPU" if "NPU" in devices else "CPU"

    ram_before = rss_mb()
    start = time.perf_counter()
    pipe = build_pipeline(MODEL_DIR, device)
    load_s = time.perf_counter() - start
    print(
        f"Завантаження на {device}: {load_s:.2f} с, "
        f"RAM процесу: {rss_mb():.0f} МБ (до завантаження {ram_before:.0f} МБ)"
    )

    stats = StreamStats(start=time.perf_counter())
    text = pipe.generate(PROMPT, max_new_tokens=100, streamer=stats.on_chunk)
    end = time.perf_counter()

    print(text)
    print("---")

    gen_s = end - stats.start
    if stats.first_token_at is None:
        print("Фрагменти не отримано, метрики недоступні.")
        return

    ttft_ms = (stats.first_token_at - stats.start) * 1000
    decode_s = end - stats.first_token_at
    tpot_ms = decode_s * 1000 / max(stats.chunks - 1, 1)
    throughput = stats.chunks / gen_s

    print(f"Фрагментів (приблизно токенів): {stats.chunks}")
    print(f"Час до першого токена (TTFT): {ttft_ms:.0f} мс")
    print(f"Час на токен (TPOT): {tpot_ms:.1f} мс")
    print(f"Швидкість: {throughput:.1f} токенів/с")
    print(f"Загальний час генерації: {gen_s * 1000:.0f} мс")


if __name__ == "__main__":
    main()
