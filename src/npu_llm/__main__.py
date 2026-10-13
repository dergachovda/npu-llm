"""Чат з LLM на Intel NPU: модель завантажується і прогрівається один раз при старті."""

import io
import sys

# Force UTF-8 output for Windows consoles to avoid UnicodeEncodeError with Cyrillic text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from .chat_service import run_turn
from .model_service import available_devices, load_pipeline, warmup
from .types import Session

EXIT_COMMANDS = {"/exit", "/quit"}
RESET_COMMAND = "/reset"


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
            user_text = input("\nYou: ").strip()
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

        print("Bot: ", end="", flush=True)
        run_turn(pipe, session, user_text)


if __name__ == "__main__":
    main()
