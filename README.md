# npu-llm

## 1. Створи середовище й встанови залежності: uv sync

## 2. Один раз експортуй модель (це може зайняти кілька хвилин)

```bash
uvx --from "optimum-intel[openvino]" --with datasets optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct --weight-format int4 models/qwen_npu

uvx --from "optimum-intel[openvino]" --with datasets optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct --weight-format int4 --sym --group-size -1 --ratio 1.0 models/qwen_npu_sym
```

## 3. Запусти: `uv run python -m npu_llm`

## 4. Перевірка кода

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```
