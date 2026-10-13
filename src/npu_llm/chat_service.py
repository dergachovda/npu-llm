import time
from .types import Pipeline, Session, StreamStats

from .config import SYSTEM_PROMPT, MAX_NEW_TOKENS

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
        session.build_prompt(user_text, SYSTEM_PROMPT),
        max_new_tokens=MAX_NEW_TOKENS,
        streamer=stats.on_chunk,
    )
    end = time.perf_counter()
    print()
    session.add("user", user_text)
    session.add("assistant", reply.strip())
    report(stats, end)
