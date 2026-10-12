"""Чат з LLM на Intel NPU: модель завантажується і прогрівається один раз при старті."""

import io
import sys
import time
from collections.abc import Callable

# Force UTF-8 output for Windows consoles to avoid UnicodeEncodeError with Cyrillic text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import openvino as ov
import openvino_genai as ov_genai
import psutil

MODEL_DIR = Path("models/qwen_npu_sym")
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
MAX_NEW_TOKENS = 200
EXIT_COMMANDS = {"/exit", "/quit"}
RESET_COMMAND = "/reset"


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
class Session:
    """Історія діалогу в форматі шаблону Qwen."""

    messages: list[tuple[str, str]] = field(default_factory=lambda: [])

    def add(self, role: str, content: str) -> None:
        """Додає повідомлення в історію.

        Args:
            role: Роль відправника, 'user' або 'assistant'.
            content: Текст повідомлення.
        """
        self.messages.append((role, content))

    def reset(self) -> None:
        """Очищає історію діалогу."""
        self.messages.clear()

    def build_prompt(self, user_text: str) -> str:
        """Збирає повний промпт з історією та новим повідомленням.

        Args:
            user_text: Нове повідомлення користувача.

        Returns:
            Рядок промпту, що закінчується заголовком відповіді асистента.
        """
        parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
        for role, content in self.messages:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)


@dataclass
class StreamStats:
    """Виводить фрагменти відповіді та збирає час для метрик."""

    start: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    chunks: int = 0

    def on_chunk(self, text: str) -> None:
        """Викликається на кожен новий фрагмент тексту.

        Args:
            text: Фрагмент відповіді.
        """
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self.chunks += 1
        print(text, end="", flush=True)


def _ignore(_text: str) -> None:
    """Порожній streamer для прогріву."""


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
    start = time.perf_counter()
    pipe.generate(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens=1,
        streamer=_ignore,
    )
    print(f"Прогрів завершено за {time.perf_counter() - start:.1f} с")


def report(stats: StreamStats, end: float) -> None:
    """Друкує метрики останньої відповіді.

    Args:
        stats: Зібрана статистика потоку.
        end: Момент завершення генерації (perf_counter).
    """
    if stats.first_token_at is None:
        print("(метрики недоступні)")
        return
    ttft_ms = (stats.first_token_at - stats.start) * 1000
    decode_s = end - stats.first_token_at
    decode_tps = (stats.chunks - 1) / decode_s if decode_s > 0 else 0.0
    print(
        f"[TTFT {ttft_ms:.0f} мс | декодування {decode_tps:.1f} токенів/с "
        f"| фрагментів {stats.chunks}]"
    )


def run_turn(pipe: Pipeline, session: Session, user_text: str) -> None:
    """Виконує один обмін повідомленнями та оновлює історію.

    Args:
        pipe: Завантажений пайплайн.
        session: Поточна історія діалогу.
        user_text: Повідомлення користувача.
    """
    stats = StreamStats()
    reply = pipe.generate(
        session.build_prompt(user_text),
        max_new_tokens=MAX_NEW_TOKENS,
        streamer=stats.on_chunk,
    )
    end = time.perf_counter()
    print()
    session.add("user", user_text)
    session.add("assistant", reply.strip())
    report(stats, end)


def main() -> None:
    """Запускає чат: завантажує модель, прогріває її і чекає повідомлень."""
    devices = available_devices()
    device = "NPU" if "NPU" in devices else "CPU"
    print(f"Доступні пристрої: {devices}, використовую {device}")

    pipe = load_pipeline(device)
    warmup(pipe)

    session = Session()
    print("Готово. Пиши повідомлення. /reset очищує історію, /exit виходить.")
    while True:
        try:
            user_text = input("\nТи: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue
        if user_text in EXIT_COMMANDS:
            break
        if user_text == RESET_COMMAND:
            session.reset()
            print("Історію очищено.")
            continue

        print("Бот: ", end="", flush=True)
        run_turn(pipe, session, user_text)


if __name__ == "__main__":
    main()
