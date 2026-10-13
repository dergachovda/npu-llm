from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


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

    def build_prompt(self, user_text: str, system_prompt: str) -> str:
        """Збирає повний промпт з історією та новим повідомленням.

        Args:
            user_text: Нове повідомлення користувача.
            system_prompt: Системний промпт.

        Returns:
            Рядок промпту, що закінчується заголовком відповіді асистента.
        """
        parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
        for role, content in self.messages:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)


@dataclass
class StreamStats:
    """Збирає фрагменти відповіді та час для метрик."""

    import time

    start: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    chunks: int = 0
    full_text: str = ""

    def on_chunk(self, text: str) -> None:
        """Збирає фрагменти відповіді.

        Args:
            text: Фрагмент відповіді.
        """
        if self.first_token_at is None:
            import time
            self.first_token_at = time.perf_counter()
        self.chunks += 1
        self.full_text += text
        # We don't print here because the animation thread is managing the line.
        # The run_turn function will handle the actual streaming print
        # if we pass the streamer a way to signal the main thread.


