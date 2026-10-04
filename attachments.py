"""Local attachment copies, document extraction, and image preparation."""

import base64
import io
import shutil
import uuid
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".log"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx"} | TEXT_EXTENSIONS
MAX_ATTACHMENTS = 5
MAX_FILE_BYTES = 25_000_000
MAX_TEXT_CHARS = 300_000
MAX_PDF_PAGES = 100
MAX_SCANNED_PAGES = 12
MAX_EMBEDDED_IMAGES = 8
MAX_IMAGE_EDGE = 1600


def attachment_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    raise ValueError(f"Формат {suffix or 'без расширения'} не поддерживается")


def validate_sources(paths):
    if not paths or len(paths) > MAX_ATTACHMENTS:
        raise ValueError(f"Выберите от 1 до {MAX_ATTACHMENTS} файлов")
    checked = []
    for source in paths:
        source = Path(source).resolve(strict=True)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Не удалось прочитать файл: {source.name}")
        if source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Файл {source.name} больше {MAX_FILE_BYTES // 1_000_000} МБ")
        checked.append((source, attachment_kind(source)))
    return checked


def copy_sources(checked, root, chat_id):
    """Copy selected files so future questions survive changes to originals."""
    root = Path(root).resolve()
    folder = root / str(int(chat_id))
    if folder.is_symlink() or folder.resolve().parent != root:
        raise ValueError("Недопустимая папка вложений чата")
    folder.mkdir(parents=True, exist_ok=True)
    copied = []
    try:
        for source, kind in checked:
            filename = uuid.uuid4().hex + source.suffix.lower()
            target = folder / filename
            shutil.copyfile(source, target)
            copied.append({
                "name": source.name[:200], "kind": kind,
                "path": str(target.relative_to(root)),
            })
        return copied
    except Exception:
        for item in copied:
            (Path(root) / item["path"]).unlink(missing_ok=True)
        raise


def stored_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Файл вложения отсутствует или имеет недопустимый путь")
    return target


def remove_chat_copies(root, chat_id):
    root = Path(root).resolve()
    folder = root / str(int(chat_id))
    if folder.is_symlink() or folder.resolve().parent != root:
        raise ValueError("Недопустимая папка вложений чата")
    if folder.is_dir():
        shutil.rmtree(folder)


def prepare_image(source, destination):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Для изображений нужен Pillow; установите зависимости приложения") from exc
    try:
        with Image.open(source) as original:
            original.verify()
        with Image.open(source) as original:
            if original.width * original.height > 40_000_000:
                raise ValueError("изображение слишком велико по разрешению")
            image = ImageOps.exif_transpose(original)
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            if image.mode != "RGB":
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            image.save(destination, "JPEG", quality=85, optimize=True)
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать изображение {Path(source).name}: {exc}") from exc
    return destination


def image_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _read_text(path):
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку {Path(path).name}")


def _pdf_text(path):
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Для PDF нужен PyMuPDF; установите зависимости приложения") from exc
    lines, scanned, illustrated = [], [], []
    omitted_illustrations = 0
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("PDF защищён паролем")
            if len(doc) > MAX_PDF_PAGES:
                raise ValueError(f"PDF содержит больше {MAX_PDF_PAGES} страниц")
            for number, page in enumerate(doc, 1):
                scale = min(1.5, MAX_IMAGE_EDGE / max(page.rect.width, page.rect.height, 1))
                text = page.get_text("text").strip()
                if text:
                    lines.append(f"[Страница {number}]\n{text}")
                    if page.get_images(full=True):
                        if len(illustrated) >= MAX_EMBEDDED_IMAGES:
                            omitted_illustrations += 1
                        else:
                            pixmap = page.get_pixmap(
                                matrix=pymupdf.Matrix(scale, scale), alpha=False
                            )
                            image_path = Path(path).with_name(
                                Path(path).stem + f"-illustrated-page-{number}.jpg"
                            )
                            pixmap.save(image_path, jpg_quality=82)
                            illustrated.append(image_path)
                else:
                    scanned.append(number)
                    if len(scanned) > MAX_SCANNED_PAGES:
                        raise ValueError(
                            f"В PDF больше {MAX_SCANNED_PAGES} страниц без текстового слоя; "
                            "разделите его на части"
                        )
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    image_path = Path(path).with_name(Path(path).stem + f"-page-{number}.jpg")
                    pixmap.save(image_path, jpg_quality=82)
                    lines.append(f"[Страница {number}: скан, текст будет распознан моделью]")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать PDF: {exc}") from exc
    return "\n\n".join(lines), scanned, illustrated, omitted_illustrations


def _docx_text(path):
    try:
        from docx import Document
        from docx.table import Table
    except ImportError as exc:
        raise RuntimeError("Для DOCX нужен python-docx; установите зависимости приложения") from exc
    document = Document(path)
    parts = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            parts.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in item.rows)
        elif item.text.strip():
            parts.append(item.text.strip())
    return "\n".join(parts)


def _xlsx_text(path):
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Для XLSX нужен openpyxl; установите зависимости приложения") from exc
    workbook = load_workbook(path, read_only=True, data_only=False)
    lines = []
    total_chars = 0
    try:
        for sheet in workbook.worksheets:
            if total_chars > MAX_TEXT_CHARS:
                break
            lines.append(f"[Лист {sheet.title}]")
            for row in sheet.iter_rows():
                values = [f"{cell.coordinate}={cell.value}" for cell in row if cell.value is not None]
                if values:
                    line = "; ".join(values)
                    lines.append(line)
                    total_chars += len(line)
                if total_chars > MAX_TEXT_CHARS:
                    break
    finally:
        workbook.close()
    return "\n".join(lines)


def _pptx_text(path):
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("Для PPTX нужен python-pptx; установите зависимости приложения") from exc
    presentation = Presentation(path)
    parts = []
    for number, slide in enumerate(presentation.slides, 1):
        parts.append(f"[Слайд {number}]")
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text.strip():
                parts.append(shape.text.strip())
            if shape.has_table:
                parts.extend(" | ".join(cell.text.strip() for cell in row.cells)
                             for row in shape.table.rows)
    return "\n".join(parts)


def _embedded_images(path):
    suffix = Path(path).suffix.lower()
    prefix = "word/media/" if suffix == ".docx" else "ppt/media/" if suffix == ".pptx" else None
    if prefix is None:
        return [], 0
    paths = []
    omitted = 0
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.startswith(prefix) and not name.endswith("/")]
        for index, member in enumerate(members):
            if len(paths) >= MAX_EMBEDDED_IMAGES:
                omitted += 1
                continue
            destination = Path(path).with_name(Path(path).stem + f"-embedded-{index}.jpg")
            try:
                if archive.getinfo(member).file_size > MAX_FILE_BYTES:
                    omitted += 1
                    continue
                prepare_image(io.BytesIO(archive.read(member)), destination)
            except ValueError:
                omitted += 1
                continue
            paths.append(destination)
    return paths, omitted


def _check_office_archive(path):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 100_000_000:
            raise ValueError("Распакованный документ превышает 100 МБ")


def process_attachment(path, kind):
    """Return extracted text and paths to images worth showing the model."""
    path = Path(path)
    if kind == "image":
        image_path = path.with_name(path.stem + "-prepared.jpg")
        prepare_image(path, image_path)
        return {"text": "", "images": [image_path], "note": ""}
    suffix = path.suffix.lower()
    if suffix in {".docx", ".xlsx", ".pptx"}:
        _check_office_archive(path)
    if suffix in TEXT_EXTENSIONS:
        text, scanned, illustrated, omitted_pdf_images = _read_text(path), [], [], 0
    elif suffix == ".pdf":
        text, scanned, illustrated, omitted_pdf_images = _pdf_text(path)
    elif suffix == ".docx":
        text, scanned, illustrated, omitted_pdf_images = _docx_text(path), [], [], 0
    elif suffix == ".xlsx":
        text, scanned, illustrated, omitted_pdf_images = _xlsx_text(path), [], [], 0
    elif suffix == ".pptx":
        text, scanned, illustrated, omitted_pdf_images = _pptx_text(path), [], [], 0
    else:
        raise ValueError(f"Формат {suffix} не поддерживается")
    note = ""
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        note = f"Текст документа ограничен первыми {MAX_TEXT_CHARS} символами."
    embedded, omitted = _embedded_images(path)
    if omitted:
        note += f" Не включено встроенных изображений: {omitted}."
    if omitted_pdf_images:
        note += f" Не включено страниц PDF с изображениями: {omitted_pdf_images}."
    image_paths = [path.with_name(path.stem + f"-page-{number}.jpg") for number in scanned]
    image_paths.extend(illustrated)
    image_paths.extend(embedded)
    text_path = path.with_name(path.stem + "-extracted.txt")
    text_path.write_text(text, encoding="utf-8")
    return {"text": text, "text_path": text_path, "images": image_paths,
            "scanned_pages": scanned, "note": note.strip()}


def chunks(text, limit=2400):
    """Split long extracted text without dropping any characters."""
    return [text[start:start + limit] for start in range(0, len(text), limit)]


def relevant_chunks(text, question, limit):
    """Select useful excerpts, retaining headings and an omission notice."""
    piece_size = max(300, min(1800, limit - 160))
    pieces = chunks(text, piece_size)
    if len(text) <= limit:
        return text
    import re
    terms = set(re.findall(r"[\wА-Яа-яЁё]{4,}", question.casefold()))
    ranked = sorted(range(len(pieces)), key=lambda index: (
        -sum(pieces[index].casefold().count(term) for term in terms), index
    ))
    chosen, used = [], 0
    for index in ranked:
        piece = pieces[index]
        if used + len(piece) > limit:
            continue
        chosen.append(index)
        used += len(piece)
    selected = "\n\n".join(f"[Фрагмент {index + 1}/{len(pieces)}]\n{pieces[index]}"
                           for index in sorted(chosen))
    return selected + f"\n\n[Показано {len(chosen)} из {len(pieces)} фрагментов документа.]"
