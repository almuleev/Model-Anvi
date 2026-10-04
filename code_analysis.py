"""Prepare complete, bounded source-code coverage for the local review mode."""

import os
from pathlib import Path

from local_tools import SKIP_DIRS, resolve_inside


CODE_SUFFIXES = {
    ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".java",
    ".kt", ".swift", ".rb", ".php", ".sh", ".ps1", ".sql",
    ".html", ".css", ".scss", ".vue", ".svelte", ".json",
    ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg",
}
CODE_NAMES = {"Dockerfile", "Makefile", "CMakeLists.txt", "Gemfile", "Procfile"}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock"}
MAX_ANALYSIS_FILES = 200
MAX_ANALYSIS_LINES = 10000
MAX_FILE_BYTES = 2_000_000
CHUNK_LINES = 250
CHUNK_CHARS = 9000
OVERLAP_LINES = 8


def _source_files(root, selected):
    if selected is not None:
        path = resolve_inside(root, selected)
        if not path.is_file():
            raise ValueError("Выбранный путь не является файлом")
        return [path]
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs
            if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or name in SKIP_NAMES:
                continue
            if path.suffix.lower() not in CODE_SUFFIXES and name not in CODE_NAMES:
                continue
            paths.append(path)
            if len(paths) > MAX_ANALYSIS_FILES:
                raise ValueError(
                    f"Найдено больше {MAX_ANALYSIS_FILES} файлов кода. "
                    "Выберите меньшую папку или один файл."
                )
    return paths


def collect_code_chunks(root, selected=None):
    """Return (chunks, file_stats); every source line occurs in at least one chunk."""
    root = Path(root).resolve(strict=True)
    paths = _source_files(root, selected)
    if not paths:
        raise ValueError("В выбранной папке не найдено файлов кода")
    chunks = []
    stats = []
    total_lines = 0
    for path in paths:
        relative = str(path.relative_to(root))
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Файл {relative} больше 2 МБ; выберите меньшую папку")
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise ValueError(f"Не удалось прочитать {relative} как UTF-8") from exc
        total_lines += len(lines)
        if total_lines > MAX_ANALYSIS_LINES:
            raise ValueError(
                f"В исходниках больше {MAX_ANALYSIS_LINES} строк. "
                "Выберите меньшую папку или один файл."
            )
        stats.append((relative, len(lines)))
        entries = []
        for line_number, line in enumerate(lines, 1):
            prefix = f"{line_number}: "
            part_size = CHUNK_CHARS - len(prefix) - 1
            for offset in range(0, len(line) or 1, part_size):
                entries.append((line_number, prefix + line[offset:offset + part_size] + "\n"))
        start = 0
        while start < len(entries):
            end = start
            size = 0
            while end < len(entries) and end - start < CHUNK_LINES:
                numbered = entries[end][1]
                if size + len(numbered) > CHUNK_CHARS:
                    break
                size += len(numbered)
                end += 1
            chunks.append({
                "path": relative,
                "start": entries[start][0],
                "end": entries[end - 1][0],
                "text": "".join(entry[1] for entry in entries[start:end]),
            })
            if end == len(entries):
                break
            start = max(start + 1, end - OVERLAP_LINES)
    return chunks, stats


def split_code_chunk(chunk):
    """Split a chunk by source lines, or a long single line by characters."""
    lines = chunk["text"].splitlines(keepends=True)
    if len(lines) < 2:
        prefix = f"{chunk['start']}: "
        content = chunk["text"].rstrip("\n")
        if not content.startswith(prefix):
            return None
        content = content[len(prefix):]
        if len(content) < 1000:
            return None
        midpoint = len(content) // 2
        return [
            {"path": chunk["path"], "start": chunk["start"],
             "end": chunk["end"], "text": prefix + content[:midpoint] + "\n"},
            {"path": chunk["path"], "start": chunk["start"],
             "end": chunk["end"], "text": prefix + content[midpoint:] + "\n"},
        ]
    midpoint = len(lines) // 2
    left = {
        "path": chunk["path"], "start": chunk["start"],
        "end": int(lines[midpoint - 1].split(":", 1)[0]),
        "text": "".join(lines[:midpoint]),
    }
    right = {
        "path": chunk["path"], "start": int(lines[midpoint].split(":", 1)[0]),
        "end": chunk["end"], "text": "".join(lines[midpoint:]),
    }
    return left, right


def source_excerpt(root, relative, line, radius=9):
    path = resolve_inside(root, relative)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start = max(0, line - 1 - radius)
    end = min(len(lines), line + radius)
    return "".join(f"{i + 1}: {lines[i]}\n" for i in range(start, end))
