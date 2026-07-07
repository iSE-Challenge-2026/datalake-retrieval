"""Corpus builders for raw data lake files and existing index artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import html
import json
import re
from html.parser import HTMLParser
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree as ET

from .text import compact, token_counts

SHEET_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
PPT_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".xml", ".json", ".jsonl", ".sql"}
TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg"}
DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt"}


@dataclass
class RetrievalChunk:
    source_path: str
    chunk_id: str
    text: str
    source_kind: str
    locator: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    text_counts: Any = field(init=False, repr=False)
    path_counts: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.text_counts = token_counts(self.text)
        self.path_counts = token_counts(self.source_path.replace("/", " "))


def build_corpus(
    data_root: Path,
    max_text_chars: int = 16000,
) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []
    root = data_root.resolve()
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            rel_path = path.relative_to(root).as_posix()
            chunks.extend(_file_chunks(path, rel_path, max_text_chars=max_text_chars))

    return _dedupe_chunks(chunks)


def build_canonical_corpus(
    canonical_dir: Path,
    text_chunk_chars: int = 1800,
    text_chunk_overlap: int = 180,
) -> list[RetrievalChunk]:
    """Build retrieval chunks from canonical texts/images/tables JSONL artifacts."""
    chunks: list[RetrievalChunk] = []
    root = canonical_dir.resolve()
    chunks.extend(_canonical_text_chunks(root / "texts.jsonl", text_chunk_chars, text_chunk_overlap))
    chunks.extend(_canonical_table_chunks(root / "tables.jsonl"))
    chunks.extend(_canonical_image_chunks(root / "images.jsonl"))
    return _dedupe_chunks(chunks)


def _canonical_text_chunks(path: Path, chunk_chars: int, overlap: int) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []
    for row in _read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        text = str(row.get("text") or "")
        if not source_path or not text:
            continue
        for index, piece in enumerate(_split_text(text, chunk_chars, overlap)):
            chunks.append(
                RetrievalChunk(
                    source_path=source_path,
                    chunk_id=str(row.get("text_id") or f"{source_path}::text::{index}"),
                    text=f"{source_path}\n{piece}",
                    source_kind="document",
                    locator=str(row.get("locator") or f"text:{index}"),
                    metadata={"canonical_type": "text", "role": row.get("role"), "parser": row.get("parser")},
                )
            )
    return chunks


def _canonical_table_chunks(path: Path) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []
    for row in _read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        preview = str(row.get("preview_text") or "")
        table_path = str(row.get("table_path") or source_path)
        if not source_path:
            continue
        text = compact(f"{source_path}\n{table_path}\n{preview}", 16000)
        chunks.append(
            RetrievalChunk(
                source_path=source_path,
                chunk_id=str(row.get("table_id") or f"{source_path}::table"),
                text=text,
                source_kind="table",
                locator=str(row.get("locator") or "table"),
                metadata={"canonical_type": "table", "role": row.get("role"), "parser": row.get("parser")},
            )
        )
    return chunks


def _canonical_image_chunks(path: Path) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []
    for row in _read_jsonl(path):
        source_path = str(row.get("source_path") or "")
        image_path = str(row.get("image_path") or source_path)
        description = str(row.get("description") or "")
        if not source_path:
            continue
        chunks.append(
            RetrievalChunk(
                source_path=source_path,
                chunk_id=str(row.get("image_id") or f"{source_path}::image"),
                text=f"{source_path}\n{image_path}\n{Path(source_path).stem.replace('_', ' ').replace('-', ' ')}\n{description}",
                source_kind="image",
                locator=str(row.get("locator") or "image"),
                metadata={"canonical_type": "image", "role": row.get("role"), "image_path": image_path},
            )
        )
    return chunks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_chars - 1))
    chunks: list[str] = []
    start = 0
    step = chunk_chars - overlap
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def _file_chunks(path: Path, rel_path: str, max_text_chars: int) -> list[RetrievalChunk]:
    suffix = path.suffix.lower()
    kind = _source_kind(suffix)
    base_text = f"{rel_path}\n{path.stem.replace('_', ' ').replace('-', ' ')}"
    extracted = ""
    try:
        if suffix in TEXT_EXTENSIONS:
            extracted = _read_text(path, max_chars=max_text_chars)
            if suffix in {".html", ".htm"}:
                extracted = _strip_html(extracted)
        elif suffix == ".csv":
            extracted = _csv_preview(path, max_rows=80)
        elif suffix == ".xlsx":
            extracted = _xlsx_preview(path, max_rows_per_sheet=60, max_shared_strings=30000)
        elif suffix == ".docx":
            extracted = _docx_text(path)
        elif suffix == ".pptx":
            extracted = _pptx_text(path)
        elif suffix == ".pdf":
            extracted = _pdf_text(path)
    except Exception as exc:
        extracted = f"Extraction failed for {rel_path}: {type(exc).__name__}: {exc}"

    text = compact(f"{base_text}\n{extracted}", max_text_chars)
    return [
        RetrievalChunk(
            source_path=rel_path,
            chunk_id=f"{rel_path}::file",
            text=text,
            source_kind=kind,
            locator="file",
            metadata={"extension": suffix},
        )
    ]


def _dedupe_chunks(chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    seen: set[tuple[str, str, str]] = set()
    output: list[RetrievalChunk] = []
    for chunk in chunks:
        key = (chunk.source_path, chunk.chunk_id, chunk.text[:200])
        if key in seen:
            continue
        seen.add(key)
        output.append(chunk)
    return output


def _source_kind(suffix: str) -> str:
    if suffix in TABLE_EXTENSIONS or suffix == ".sql":
        return "table"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in DOC_EXTENSIONS or suffix in TEXT_EXTENSIONS:
        return "document"
    return "file"


def _read_text(path: Path, max_chars: int) -> str:
    data = path.read_bytes()[: max_chars * 4]
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return data.decode(encoding)[:max_chars]
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")[:max_chars]


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)


def _strip_html(text: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(text)
    return html.unescape(" ".join(parser.parts))


def _csv_preview(path: Path, max_rows: int) -> str:
    text = _read_text(path, max_chars=120000)
    try:
        dialect = csv.Sniffer().sniff(text[:4096])
    except csv.Error:
        dialect = csv.excel
    rows = []
    for index, row in enumerate(csv.reader(text.splitlines(), dialect)):
        if index >= max_rows:
            break
        rows.append("\t".join(row))
    return "\n".join(rows)


def _xlsx_preview(path: Path, max_rows_per_sheet: int, max_shared_strings: int) -> str:
    if path.stat().st_size > 30_000_000:
        return f"Large workbook {path.name}; skipped detailed preview."
    with ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive, max_items=max_shared_strings)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pkgrel:Relationship", SHEET_NS)}
        parts: list[str] = []
        for sheet in workbook.findall("main:sheets/main:sheet", SHEET_NS):
            sheet_name = sheet.attrib.get("name", "Sheet")
            rel_id = sheet.attrib.get(f"{{{SHEET_NS['rel']}}}id", "")
            target = relmap.get(rel_id, "")
            if not target.startswith("/"):
                target = "xl/" + target.lstrip("/")
            target = target.replace("xl/xl/", "xl/")
            rows = _xlsx_sheet_rows(archive, target, shared, max_rows=max_rows_per_sheet)
            if rows:
                parts.append(f"Sheet {sheet_name}\n" + "\n".join("\t".join(row) for row in rows))
        return "\n\n".join(parts)


def _xlsx_shared_strings(archive: ZipFile, max_items: int) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    strings: list[str] = []
    with archive.open("xl/sharedStrings.xml") as handle:
        for event, element in ET.iterparse(handle, events=("end",)):
            if element.tag.endswith("}si"):
                strings.append("".join(text.text or "" for text in element.iter() if text.tag.endswith("}t")))
                element.clear()
                if len(strings) >= max_items:
                    break
    return strings


def _xlsx_sheet_rows(archive: ZipFile, target: str, shared: list[str], max_rows: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with archive.open(target) as handle:
        for event, element in ET.iterparse(handle, events=("end",)):
            if not element.tag.endswith("}row"):
                continue
            values: list[str] = []
            for cell in list(element):
                if not cell.tag.endswith("}c"):
                    continue
                index = _col_to_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                values[index] = _xlsx_cell_value(cell, shared)
            if any(values):
                rows.append(values)
            element.clear()
            if len(rows) >= max_rows:
                break
    return rows


def _col_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "A1")
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _xlsx_cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = next((child.text or "" for child in list(cell) if child.tag.endswith("}v")), "")
    if cell.attrib.get("t") == "s" and value:
        index = int(value)
        return shared[index] if index < len(shared) else ""
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.iter() if text.tag.endswith("}t"))
    return value


def _docx_text(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
            return " ".join(text.text or "" for text in root.findall(".//w:t", WORD_NS))
    except (BadZipFile, KeyError, ET.ParseError):
        return ""


def _pptx_text(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            slide_names = sorted(
                (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)),
            )
            parts = []
            for index, name in enumerate(slide_names, start=1):
                root = ET.fromstring(archive.read(name))
                text = " ".join(item.text or "" for item in root.findall(".//a:t", PPT_NS))
                if text:
                    parts.append(f"Slide {index}: {text}")
            return "\n\n".join(parts)
    except (BadZipFile, KeyError, ET.ParseError, AttributeError):
        return ""


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages[:12])
    except Exception:
        return ""

