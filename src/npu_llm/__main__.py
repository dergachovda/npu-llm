"""Перевірка NPU та запуск LLM на ньому."""

import sys
from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai

MODEL_DIR = Path("qwen_npu")


def available_devices() -> list[str]:
    """Повертає пристрої, які бачить OpenVINO.

    Returns:
        Назви пристроїв, наприклад ['CPU', 'GPU', 'NPU'].
    """
    return list(ov.Core().available_devices)


def build_pipeline(model_dir: Path, device: str) -> ov_genai.LLMPipeline:
    """Створює LLM-пайплайн на вказаному пристрої.

    Args:
        model_dir: Каталог з експортованою моделлю OpenVINO.
        device: Назва пристрою, наприклад 'NPU' або 'CPU'.

    Returns:
        Готовий до генерації пайплайн.
    """
    return ov_genai.LLMPipeline(str(model_dir), device)


def main() -> None:
    """Запускає генерацію на NPU, а якщо його немає, то на CPU."""
    devices = available_devices()
    print(f"Доступні пристрої: {devices}")

    if not MODEL_DIR.exists():
        sys.exit("Модель не знайдено. Спочатку виконай експорт (крок 3 нижче).")

    device = "NPU" if "NPU" in devices else "CPU"
    pipe = build_pipeline(MODEL_DIR, device)
    print(pipe.generate("Привіт! Що таке NPU?", max_new_tokens=100))


if __name__ == "__main__":
    main()
