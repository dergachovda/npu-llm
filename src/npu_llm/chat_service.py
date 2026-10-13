import time

from .config import MAX_NEW_TOKENS, SYSTEM_PROMPT
from .types import Pipeline, Session, StreamStats


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
    import itertools
    import sys
    from threading import Event, Thread

    stats = StreamStats()
    # Animation for bot thinking
    animation = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠴", "⠼", "⠸", "⠹", "⠙", "⠋"])

    stop_anim = Event()

    def animate():
        while not stop_anim.is_set():
            if not stats.full_text:
                sys.stdout.write(f"\rBot: {next(animation)}")
                sys.stdout.flush()
            else:
                stop_anim.set()
            time.sleep(0.1)

    anim_thread = Thread(target=animate, daemon=True)
    anim_thread.start()

    def streaming_handler(text: str) -> None:
        # If this is the first token, we clear the animation line once
        if stats.chunks == 0:
            sys.stdout.write("\rBot: ")
            sys.stdout.flush()

        # Update stats and print the token immediately
        stats.on_chunk(text)
        sys.stdout.write(text)
        sys.stdout.flush()

        # Signal animation to stop
        stop_anim.set()

    try:
        reply = pipe.generate(
            session.build_prompt(user_text, SYSTEM_PROMPT),
            max_new_tokens=MAX_NEW_TOKENS,
            streamer=streaming_handler,
        )
    finally:
        stop_anim.set()
        anim_thread.join()

    end = time.perf_counter()
    print()
    session.add("user", user_text)
    session.add("assistant", reply.strip())
    report(stats, end)
