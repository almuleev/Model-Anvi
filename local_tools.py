"""Read-only workspace tools exposed to the local model."""

from pathlib import Path


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
MAX_FILES = 2000
MAX_FILE_BYTES = 1_000_000
MAX_RESULT_CHARS = 1800


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List relative file paths in the selected workspace. Call before reading an unknown file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative subdirectory, or empty for workspace root"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search text in workspace files and return matching paths and line numbers.",
            "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string", "description": "Literal text to find"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded range of lines from one workspace file. Use start_line to continue reading long files.",
            "parameters": {"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string", "description": "Relative file path from workspace root"},
                "start_line": {"type": "integer", "description": "First line, starting at 1"},
            }},
        },
    },
]


def resolve_inside(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Путь должен быть строкой")
    root = Path(root).resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    if candidate != root and root not in candidate.parents:
        raise ValueError("Путь находится вне выбранной папки")
    return candidate


def iter_files(root):
    count = 0
    for directory, dirs, files in __import__("os").walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            count += 1
            if count > MAX_FILES:
                return
            yield path


def execute_tool(root, name, arguments):
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Папка проекта недоступна")
        if not isinstance(arguments, dict):
            raise ValueError("Неверные аргументы инструмента")
        if name == "list_files":
            target = resolve_inside(root, arguments.get("path", ""))
            if not target.is_dir():
                raise ValueError("Указан не каталог")
            paths = []
            total = 0
            for path in iter_files(target):
                relative = str(path.relative_to(root))
                if total + len(relative) > MAX_RESULT_CHARS or len(paths) == 100:
                    return "\n".join(paths) + "\n[Список сокращён; укажите подкаталог]"
                paths.append(relative)
                total += len(relative) + 1
            return "\n".join(paths) or "Файлы не найдены"
        if name == "search_text":
            query = arguments.get("query", "")
            if not isinstance(query, str) or not 1 <= len(query) <= 200:
                raise ValueError("Поисковый запрос должен содержать от 1 до 200 символов")
            found = []
            total = 0
            for path in iter_files(root):
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                try:
                    with path.open("r", encoding="utf-8") as source:
                        for line_no, line in enumerate(source, 1):
                            if query.casefold() in line.casefold():
                                match = f"{path.relative_to(root)}:{line_no}: {line.rstrip()[:180]}"
                                if total + len(match) > MAX_RESULT_CHARS or len(found) == 30:
                                    return "\n".join(found) + "\n[Результаты сокращены; уточните запрос]"
                                found.append(match)
                                total += len(match) + 1
                except (UnicodeError, OSError):
                    continue
            return "\n".join(found) or "Совпадений нет"
        if name == "read_file":
            path = resolve_inside(root, arguments.get("path", ""))
            if not path.is_file():
                raise ValueError("Указан не файл")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Файл больше 1 МБ; используйте поиск или более узкий файл")
            start = arguments.get("start_line", 1)
            if type(start) is not int or start < 1:
                raise ValueError("start_line должен быть положительным целым числом")
            lines = []
            total = 0
            with path.open("r", encoding="utf-8") as source:
                for line_no, line in enumerate(source, 1):
                    if line_no < start:
                        continue
                    content = line.rstrip("\r\n")
                    if len(content) > 400:
                        content = content[:400] + " [строка сокращена]"
                    entry = f"{line_no}: {content}"
                    if len(lines) == 100 or total + len(entry) > MAX_RESULT_CHARS:
                        lines.append(f"[Далее: read_file(path=\"{path.relative_to(root)}\", start_line={line_no})]")
                        break
                    lines.append(entry)
                    total += len(entry) + 1
            return "\n".join(lines) or "Строк больше нет"
        return "Неизвестный инструмент"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"Ошибка инструмента: {exc}"
