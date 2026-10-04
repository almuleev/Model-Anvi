"""Remove data created by the local chat after its window has closed."""

import json
import shutil
import sqlite3
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


APP_DIR = Path(__file__).resolve().parent
MODEL = "qwen3.8:27b-q4_K_M"
OPENER = build_opener(ProxyHandler({}))


def unload_local_model():
    """Drop the local Ollama context if its HTTP server is still available."""
    try:
        with OPENER.open("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            loaded = json.load(response).get("models", [])
    except HTTPError as exc:
        raise RuntimeError(f"Ollama не подтвердила состояние модели: HTTP {exc.code}") from exc
    except (OSError, URLError):
        return
    if not any(item.get("name") == MODEL or item.get("model") == MODEL
               for item in loaded):
        return
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": MODEL, "keep_alive": 0, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with OPENER.open(request, timeout=30) as response:
            response.read()
        with OPENER.open("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            loaded = json.load(response).get("models", [])
    except (OSError, URLError) as exc:
        raise RuntimeError("Не удалось подтвердить выгрузку модели из памяти") from exc
    if any(item.get("name") == MODEL or item.get("model") == MODEL
           for item in loaded):
        raise RuntimeError("Модель всё ещё загружена в Ollama")


def purge_local_data(app_dir=APP_DIR):
    """Erase app-owned SQLite data and remove its database and server log.

    Call only after the chat has closed all database connections and stopped
    its own Ollama process. User-selected source files are never modified.
    """
    app_dir = Path(app_dir)
    database_paths = [app_dir / "history.sqlite3", *sorted(app_dir.glob("markdown-test-*.sqlite3"))]
    chat_count = message_count = 0
    for db_path in database_paths:
        if not db_path.exists():
            continue
        if db_path.is_symlink():
            raise RuntimeError(f"База данных является ссылкой: {db_path.name}")
        db_uri = db_path.resolve().as_uri() + "?mode=rw"
        db = sqlite3.connect(db_uri, uri=True, timeout=2)
        try:
            db.execute("PRAGMA secure_delete = ON")
            db.execute("PRAGMA journal_mode = DELETE")
            db.execute("BEGIN EXCLUSIVE")
            chat_count += db.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            message_count += db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            db.execute("DELETE FROM messages")
            db.execute("DELETE FROM chats")
            db.execute("DELETE FROM settings")
            db.commit()
            db.execute("VACUUM")
            if any(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                   for table in ("chats", "messages", "settings")):
                raise RuntimeError("База данных не очистилась полностью")
        finally:
            db.close()
    paths = [app_dir / "server.log"]
    for db_path in database_paths:
        paths.extend((db_path, *(Path(str(db_path) + suffix)
                                 for suffix in ("-wal", "-shm", "-journal"))))
    for path in paths:
        path.unlink(missing_ok=True)
    attachments_dir = app_dir / "attachments"
    if (attachments_dir.is_symlink()
            or attachments_dir.resolve().parent != app_dir.resolve()):
        raise RuntimeError("Папка вложений имеет недопустимый путь")
    if attachments_dir.is_dir():
        shutil.rmtree(attachments_dir)
    if any(path.exists() for path in paths):
        raise RuntimeError("Некоторые файлы приложения не удалились")
    if attachments_dir.exists():
        raise RuntimeError("Некоторые вложения не удалились")
    return chat_count, message_count


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        launched_from_chat = "--after-close" in sys.argv
        if not launched_from_chat and not messagebox.askyesno(
            "Удалить все данные чата",
            "Сначала закройте окно чата. Удалить все переписки, настройки и журнал "
            "приложения? Исходные файлы проекта останутся на месте.\n\n"
            "Действие нельзя отменить.", parent=root,
        ):
            return
        if launched_from_chat:
            time.sleep(1)
        memory_error = None
        try:
            unload_local_model()
        except RuntimeError as exc:
            memory_error = str(exc)
        for attempt in range(60 if launched_from_chat else 1):
            try:
                chats, messages = purge_local_data()
                break
            except (OSError, sqlite3.Error):
                if not launched_from_chat or attempt == 59:
                    raise
                time.sleep(0.5)
        if memory_error:
            raise RuntimeError(
                f"Данные на диске удалены, но {memory_error}. "
                "Закройте Ollama и повторите очистку памяти."
            )
        messagebox.showinfo(
            "Данные приложения удалены",
            f"Удалено чатов: {chats}. Сообщений: {messages}.\n\n"
            "База, настройки и локальный журнал удалены. "
            "Исходные файлы проекта не изменены.\n\n"
            "Копии в резервных хранилищах и физическое восстановление с SSD "
            "приложение исключить не может.", parent=root,
        )
    except Exception as exc:
        messagebox.showerror(
            "Ошибка очистки",
            f"Не удалось полностью очистить данные приложения:\n{exc}", parent=root,
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
