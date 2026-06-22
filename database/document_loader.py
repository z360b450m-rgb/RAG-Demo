from pathlib import Path
from typing import Callable, Dict


def load_pdf(file_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def load_txt(file_path: Path) -> str:
    raw = file_path.read_bytes()
    for encoding in ("utf-8", "gb2312", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def load_markdown(file_path: Path) -> str:
    return load_txt(file_path)


SUPPORTED_EXTENSIONS: Dict[str, Callable[[Path], str]] = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_markdown,
}


def load_document(file_path: str | Path) -> tuple[str, str]:
    """Load a document and return (filename, text_content)."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    loader = SUPPORTED_EXTENSIONS[suffix]
    text = loader(file_path)
    if not text.strip():
        raise ValueError(f"No text extracted from {file_path.name}")
    return file_path.name, text.strip()
