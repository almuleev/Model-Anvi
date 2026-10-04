"""Lightweight per-answer timing and token accounting for the local chat."""

import json
import time
from collections import OrderedDict


STAGE_NAMES = {
    "chat": "Модель / чат",
    "attachments": "Подготовка вложений",
    "prepare": "Подготовка кода",
    "scan": "Первичный просмотр",
    "review": "Повторная проверка",
    "compress": "Сжатие сводок",
    "final": "Итоговый ответ",
    "tools": "Чтение файлов",
}


def _seconds(nanoseconds):
    return max(0.0, float(nanoseconds or 0) / 1_000_000_000)


class StreamMeter:
    """Observe streamed fields without tokenizing them during generation."""

    def __init__(self):
        self.phase = None
        self.phase_started = None
        self.thinking_time = 0.0
        self.answer_time = 0.0
        self.thinking_chars = 0
        self.answer_chars = 0
        self.tool_chars = 0

    def _enter(self, phase):
        now = time.perf_counter()
        if phase == self.phase:
            return
        if self.phase == "thinking":
            self.thinking_time += now - self.phase_started
        elif self.phase == "answer":
            self.answer_time += now - self.phase_started
        self.phase = phase
        self.phase_started = now

    def observe(self, message):
        if message.get("thinking"):
            self._enter("thinking")
            self.thinking_chars += len(message["thinking"])
        if message.get("content"):
            self._enter("answer")
            self.answer_chars += len(message["content"])
        if message.get("tool_calls"):
            self.tool_chars += len(json.dumps(message["tool_calls"], ensure_ascii=False))

    def finish(self):
        self._enter(None)
        return {
            "thinking_chars": self.thinking_chars,
            "answer_chars": self.answer_chars,
            "tool_chars": self.tool_chars,
            "thinking_time": self.thinking_time,
            "answer_time": self.answer_time,
        }


class UsageReport:
    def __init__(self, preset, mode):
        self.preset = preset
        self.mode = mode
        self.started = time.perf_counter()
        self.finished = None
        self.current = "chat" if mode == "chat" else "prepare"
        self.stage_started = self.started
        self.stages = OrderedDict()
        self._stage(self.current)
        self.incomplete = False

    def _stage(self, name):
        return self.stages.setdefault(name, {
            "wall": 0.0, "requests": 0, "input": 0, "output": 0,
            "missing": 0, "load": 0.0, "prompt": 0.0, "generate": 0.0,
            "thinking_time": 0.0, "answer_time": 0.0,
            "thinking_chars": 0, "answer_chars": 0, "tool_chars": 0,
        })

    def switch(self, name):
        now = time.perf_counter()
        if self.finished is not None or name == self.current:
            return
        self._stage(self.current)["wall"] += now - self.stage_started
        self.current = name
        self.stage_started = now
        self._stage(name)

    def add_request(self, stage, final_event, thinking_chars=0, answer_chars=0,
                    tool_chars=0, thinking_time=0.0, answer_time=0.0):
        row = self._stage(stage)
        row["requests"] += 1
        if final_event is None or "prompt_eval_count" not in final_event or "eval_count" not in final_event:
            row["missing"] += 1
            self.incomplete = True
        else:
            row["input"] += int(final_event["prompt_eval_count"])
            row["output"] += int(final_event["eval_count"])
            row["load"] += _seconds(final_event.get("load_duration"))
            row["prompt"] += _seconds(final_event.get("prompt_eval_duration"))
            row["generate"] += _seconds(final_event.get("eval_duration"))
        if final_event is not None and "eval_count" in final_event:
            row["thinking_chars"] += thinking_chars
            row["answer_chars"] += answer_chars
            row["tool_chars"] += tool_chars
        row["thinking_time"] += thinking_time
        row["answer_time"] += answer_time

    def finish(self):
        if self.finished is None:
            self.finished = time.perf_counter()
            self._stage(self.current)["wall"] += self.finished - self.stage_started

    def as_dict(self):
        self.finish()
        return {
            "preset": self.preset, "mode": self.mode,
            "total_time": self.finished - self.started,
            "incomplete": self.incomplete,
            "stages": self.stages,
        }


def _duration(value):
    return f"{value:.2f} с"


def format_report(data):
    """Render fixed-width text; approximate splits never replace exact totals."""
    rows = data["stages"]
    totals = {key: sum(row[key] for row in rows.values()) for key in (
        "requests", "input", "output", "missing", "load", "prompt", "generate",
        "thinking_time", "answer_time", "thinking_chars", "answer_chars", "tool_chars",
    )}
    lines = [f"Отчёт · {data['preset']} · всего {_duration(data['total_time'])}",
             "Этап                     Время     Запросы   Вход    Выход"]
    for name, row in rows.items():
        if name == "tools" and not row["wall"]:
            continue
        if name == "tools":
            lines.append(f"{STAGE_NAMES[name]:<24} {_duration(row['wall']):>8} {'—':>7} {'—':>7} {'—':>8}")
            continue
        token_suffix = "+?" if row["missing"] else ""
        lines.append(
            f"{STAGE_NAMES.get(name, name):<24} {_duration(row['wall']):>8} "
            f"{row['requests']:>7} {str(row['input']) + token_suffix:>7} "
            f"{str(row['output']) + token_suffix:>8}"
        )
    suffix = "+?" if totals["missing"] else ""
    lines.append(
        f"{'Всего':<24} {_duration(data['total_time']):>8} "
        f"{totals['requests']:>7} {str(totals['input']) + suffix:>7} "
        f"{str(totals['output']) + suffix:>8}"
    )
    lines.append(
        "Ollama: загрузка " + _duration(totals["load"]) +
        ", вход " + _duration(totals["prompt"]) +
        ", генерация " + _duration(totals["generate"])
    )
    if totals["thinking_chars"] or totals["answer_chars"] or totals["tool_chars"]:
        char_total = totals["thinking_chars"] + totals["answer_chars"] + totals["tool_chars"]
        approx_thinking = round(totals["output"] * totals["thinking_chars"] / char_total)
        approx_tools = round(totals["output"] * totals["tool_chars"] / char_total)
        approx_answer = totals["output"] - approx_thinking - approx_tools
        lines.append(
            "Фазы потока: размышление " + _duration(totals["thinking_time"]) +
            f" (≈{approx_thinking} ток.), ответ " + _duration(totals["answer_time"]) +
            f" (≈{approx_answer} ток.), вызовы инструментов ≈{approx_tools} ток."
        )
        lines.append("≈ — оценка по доле символов; точный счётчик есть только для всего запроса.")
    if totals["missing"]:
        lines.append(f"Нет финальных счётчиков для {totals['missing']} запросов; итог токенов неполный.")
    elif data.get("incomplete"):
        lines.append("Ответ прерван или не завершён; счётчики могут быть неполными.")
    return "\n".join(lines)


def serialize_report(report):
    return json.dumps(report.as_dict(), ensure_ascii=False)
