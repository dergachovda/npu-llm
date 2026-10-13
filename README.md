# npu-llm

Проєкт для запуску LLM на NPU (Neural Processing Unit).

> **⚠️ Важливо:** На даний момент проєкт підтримує та оптимізований виключно для **Intel NPU** (використовуючи OpenVINO).

## 🚀 Швидкий старт (Getting Started)


Цей розділ допоможе вам запустити проєкт з нуля.

### 1. Пререквізити
Переконайтеся, що у вас встановлені:
- **Python 3.14** (експериментальна версія)
- **uv** (швидкий менеджер пакетів Python)

### 2. Встановлення
Клонуйте репозиторій та встановіть залежності:
```bash
git clone <url-репозиторію>
cd npu-llm
uv sync
```

### 3. Підготовка моделі
Один раз експортуйте модель (це може зайняти кілька хвилин):
```bash
uvx --from "optimum-intel[openvino]" --with datasets optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct --weight-format int4 --sym --group-size -1 --ratio 1.0 models/qwen_npu_sym
```

### 4. Запуск застосунку
Запустіть чат-застосунок за допомогою `uv`:
```bash
uv run npu-llm
```

---

## 🛠 Якість коду та валідація

Для перевірки проєкту на відповідність стандартам (лінтинг, форматування, статичний аналіз) використовуйте команду:
```bash
uv run validation
```

Або запустіть окремі інструменти:
```bash
uv run ruff check             # Перевірка стилю
uv run ruff format --check    # Перевірка форматування
uv run pyright               # Статичний аналіз типів
uv run pytest                # Тести
```

