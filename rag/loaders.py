"""File loaders — reads .txt/.md/.pdf into (source_name, text) tuples."""

import os


def load_file(path: str) -> tuple[str, str]:
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:  # .txt, .md, or anything else — read as plain text
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return name, text


def load_files(paths: list[str]) -> list[tuple[str, str]]:
    return [load_file(p) for p in paths]
