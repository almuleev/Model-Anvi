r"""Render the real UI with authored examples in an isolated, temporary database.

This documentation preview never contacts Ollama or opens the user's history.
Run from the repository: .venv\Scripts\python.exe tools/preview_docs.py --scene chat
"""

import argparse
import importlib.util
import sys
import tempfile
import tkinter as tk
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
loader = SourceFileLoader("neuroquay_preview_app", str(ROOT / "local_chat.pyw"))
spec = importlib.util.spec_from_loader(loader.name, loader)
app_module = importlib.util.module_from_spec(spec)
loader.exec_module(app_module)

# SQLite's own context manager commits/rolls back but does not close the file.
# Close preview connections deterministically before cleaning the temporary dir.
original_db_connect = app_module.db_connect


@contextmanager
def preview_db_connect():
    connection = original_db_connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


app_module.db_connect = preview_db_connect

EXAMPLES = [
    (
        "Как начать с NeuroQuay",
        "Для каких задач использовать NeuroQuay?",
        "## Твой локальный ИИ для файлов и кода\n\n"
        "**Чат.** Задавай вопросы, уточняй ответ и возвращайся к сохранённым диалогам.\n\n"
        "**Документы и изображения.** Прикрепи PDF, таблицу или фото и напиши, "
        "что нужно найти или объяснить.\n\n"
        "**Код.** Выбери папку проекта для чтения файлов. Кнопки «Анализ файла» "
        "и «Анализ папки» запускают последовательный обзор исходников.\n\n"
        "Начни с короткого вопроса. Глубину рассуждения можно менять над диалогом.",
    ),
    (
        "План проекта из документа",
        "Выдели три задачи из описания проекта.",
        "## План действий\n\n"
        "1. Описать запуск приложения и настройку Ollama.\n"
        "2. Подготовить карту модулей и поток обработки запросов.\n"
        "3. Добавить тесты в Windows CI и снимки интерфейса.\n\n"
        "После этого проверь запуск на чистом виртуальном окружении.",
    ),
    (
        "Обзор кода: average.py",
        "Анализ кода файла average.py: проверь обработку пустого списка.",
        "## Проверка average.py\n\n"
        "**average.py:2 — деление на ноль для пустого списка.**\n\n"
        "`sum(values) / len(values)` при `values=[]` вызывает `ZeroDivisionError`. "
        "Сначала проверь длину списка и явно сообщи о некорректном аргументе.\n\n"
        "```python\n"
        "def average(values):\n"
        "    if not values:\n"
        "        raise ValueError(\"Список пуст\")\n"
        "    return sum(values) / len(values)\n"
        "```\n\n"
        "Добавь проверки для пустого списка и для нескольких чисел. "
        "Исходный файл остаётся под твоим управлением.",
    ),
]


class PreviewChat(app_module.LocalChat):
    """Reuse the UI, with network and destructive actions disabled for preview."""

    def ensure_server(self):
        pass

    def send(self, _event=None):
        return "break"

    def start_code_analysis(self, relative_file):
        pass

    def stop_model(self):
        pass

    def delete_chat(self):
        pass

    def delete_all_data(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("chat", "code"), default="chat")
    parser.add_argument("--seconds", type=int, default=0, help="Auto-close after N seconds")
    parser.add_argument("--output", type=Path, help="Save this demo window as PNG and exit (Windows)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="neuroquay-docs-") as temporary:
        preview_dir = Path(temporary)
        app_module.APP_DIR = preview_dir
        app_module.DB_PATH = preview_dir / "history.sqlite3"
        app_module.ATTACHMENTS_DIR = preview_dir / "attachments"
        app_module.init_db()
        with app_module.db_connect() as db:
            for number, (title, question, answer) in enumerate(EXAMPLES, 1):
                db.execute(
                    "INSERT INTO chats(id, title, updated_at) VALUES (?, ?, ?)",
                    (number, title, f"2026-10-04T12:0{4-number}:00"),
                )
                db.executemany(
                    "INSERT INTO messages(chat_id, role, content) VALUES (?, ?, ?)",
                    ((number, "user", question), (number, "assistant", answer)),
                )
            db.execute("INSERT INTO settings VALUES ('reasoning_preset', '4x')")
            db.execute("INSERT INTO settings VALUES ('composer_height', '220')")

        root = tk.Tk()
        ui = PreviewChat(root)
        root.title("NeuroQuay · Qwen3.8 · Демо")
        root.geometry("1260x840" if args.scene == "chat" else "1260x940")

        def show_scene():
            ui.ready = True
            ui.set_action_buttons("normal")
            ui.stop_button.configure(state="normal")
            ui.set_status("Демонстрация интерфейса · ответы в примере подготовлены заранее")
            ui.chat_id = 1 if args.scene == "chat" else 3
            ui.refresh_chats()
            ui.show_chat()
            ui.transcript.yview_moveto(0)
            ui.editor_panes.sash_place(0, 0, 430 if args.scene == "chat" else 530)
            if args.scene == "chat":
                sample = preview_dir / "Описание проекта.txt"
                sample.write_text("NeuroQuay: локальный чат, документы и код.", encoding="utf-8")
                ui.pending_attachments = [sample]
                ui.refresh_attachment_list()
                ui.prompt.insert("1.0", "Составь короткий план по прикреплённому описанию проекта.")
            else:
                source = preview_dir / "demo-project"
                source.mkdir()
                (source / "average.py").write_text(
                    "def average(values):\n    return sum(values) / len(values)\n", encoding="utf-8"
                )
                ui.workspace = source
                ui.workspace_label.configure(text="Папка: demo-project (пример)")
                ui.prompt.insert("1.0", "Проверь также обработку нечисловых значений.")
                ui.set_action_buttons("normal")
            ui.prompt.focus_set()

        root.after(200, show_scene)
        capture_errors = []
        if args.output:
            def capture():
                try:
                    import ctypes
                    from ctypes import wintypes
                    from PIL import ImageGrab

                    root.update_idletasks()
                    get_ancestor = ctypes.windll.user32.GetAncestor
                    get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
                    get_ancestor.restype = wintypes.HWND
                    handle = get_ancestor(root.winfo_id(), 2)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    ImageGrab.grab(window=handle).save(args.output, format="PNG")
                    print(f"Saved {args.output.resolve()}")
                except Exception as error:
                    capture_errors.append(error)
                finally:
                    root.destroy()
            root.after(1200, capture)
        if args.seconds > 0:
            root.after(args.seconds * 1000, root.destroy)
        root.mainloop()
        if capture_errors:
            raise RuntimeError("Could not capture the documentation preview") from capture_errors[0]


if __name__ == "__main__":
    main()
