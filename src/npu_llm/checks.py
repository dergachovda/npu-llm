"""Запуск усіх перевірок якості коду однією командою."""

import io
import subprocess
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

COMMANDS: list[list[str]] = [
    ["ruff", "check", "--fix"],
    ["ruff", "format"],
    ["pyright"],
]


def validation() -> None:
    """Виконує ruff check, ruff format та pyright по черзі.

    Зупиняється на першій команді, яка повернула помилку.
    """
    for cmd in COMMANDS:
        print(f"> {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("Усі перевірки пройдені.")


if __name__ == "__main__":
    validation()
