"""Question readers for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import csv
import json
import re

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class RetrievalQuestion:
    question_id: str
    question: str
    data_sources: list[str] = field(default_factory=list)
    answer_type: str = ""


def read_questions(path: Path) -> list[RetrievalQuestion]:
    if path.suffix.lower() == ".xlsx":
        return _read_xlsx(path)
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    raise ValueError(f"Unsupported question file: {path}")


def _read_csv(path: Path) -> list[RetrievalQuestion]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        output: list[RetrievalQuestion] = []
        for index, row in enumerate(reader, start=1):
            norm = {_norm_header(key): value for key, value in row.items()}
            question = (norm.get("question") or "").strip()
            if not question:
                continue
            output.append(
                RetrievalQuestion(
                    question_id=_clean_id(norm.get("id") or norm.get("stt") or index),
                    question=question,
                    data_sources=_parse_sources(norm.get("data_sources", "")),
                    answer_type=norm.get("answer_type", ""),
                )
            )
        return output


def _read_xlsx(path: Path) -> list[RetrievalQuestion]:
    rows = _xlsx_rows(path)
    if not rows:
        return []
    headers = [_norm_header(value) for value in rows[0]]
    output: list[RetrievalQuestion] = []
    for row in rows[1:]:
        values = {headers[index]: row[index] if index < len(row) else "" for index in range(len(headers))}
        question = values.get("question", "").strip()
        if not question:
            continue
        output.append(
            RetrievalQuestion(
                question_id=_clean_id(values.get("stt") or values.get("id") or len(output) + 1),
                question=question,
                data_sources=_parse_sources(values.get("data_sources", "")),
                answer_type=values.get("answer_type", ""),
            )
        )
    return output


def _xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as archive:
        shared = _shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pkgrel:Relationship", NS)}
        first_sheet = workbook.find("main:sheets/main:sheet", NS)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get(f"{{{NS['rel']}}}id", "")
        target = relmap.get(rel_id, "")
        if not target.startswith("/"):
            target = "xl/" + target.lstrip("/")
        target = target.replace("xl/xl/", "xl/")
        root = ET.fromstring(archive.read(target))
        rows: list[list[str]] = []
        for row in root.findall(".//main:sheetData/main:row", NS):
            values: list[str] = []
            for cell in row.findall("main:c", NS):
                index = _col_to_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                values[index] = _cell_value(cell, shared)
            if any(values):
                rows.append(values)
        return rows


def _shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(text.text or "" for text in item.findall(".//main:t", NS)) for item in root.findall("main:si", NS)]


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.find("main:v", NS)
    if cell.attrib.get("t") == "s" and value is not None:
        index = int(value.text or 0)
        return shared[index] if index < len(shared) else ""
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", NS))
    return value.text if value is not None else ""


def _col_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "A1")
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _parse_sources(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass
    return [text]


def _norm_header(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _clean_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text
