import sys
import time
from pathlib import Path
from typing import cast

import openvino as ov
import openvino_genai as ov_genai
import psutil

from .config import MODEL_DIR
from .types import Pipeline


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


def load_pipeline(device: str) -> Pipeline:
    """Завантажує модель і показує час та пам'ять.

    Args:
        device: Пристрій для запуску моделі.

    Returns:
        Завантажений пайплайн.
    """
    if not MODEL_DIR.exists():
        sys.exit(f"Модель не знайдено: {MODEL_DIR}")

    ram_before = rss_mb()
    start = time.perf_counter()
    pipe = build_pipeline(MODEL_DIR, device)
    load_s = time.perf_counter() - start
    print(
        f"Модель завантажена на {device} за {load_s:.1f} с, "
        f"RAM {rss_mb():.0f} МБ (було {ram_before:.0f} МБ)"
    )
    return pipe


def warmup(pipe: Pipeline) -> None:
    """Прогріває пайплайн коротким запитом, щоб перший реальний запит був швидшим.

    Args:
        pipe: Завантажений пайплайн.
    """

    def _ignore(_text: str) -> None:
        pass

    start = time.perf_counter()
    pipe.generate(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens=1,
        streamer=_ignore,
    )
    print(f"Прогрів завершено за {time.perf_counter() - start:.1f} с")
