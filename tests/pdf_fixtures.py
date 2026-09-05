"""Build tiny valid PDFs in memory for tests. No course materials."""

from __future__ import annotations


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def minimal_pdf_bytes(page_texts: list[str]) -> bytes:
    """Return a minimal PDF-1.4 document with one page per string."""
    if not page_texts:
        raise ValueError("page_texts must not be empty")

    n = len(page_texts)
    font_id = 3 + 2 * n
    objects: dict[int, bytes] = {}

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{i} 0 R" for i in range(3, 3 + n))
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("ascii")

    for i, text in enumerate(page_texts):
        page_id = 3 + i
        content_id = 3 + n + i
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")
        stream = f"BT /F1 24 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET"
        stream_bytes = stream.encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    chunks: list[bytes] = [header]
    offsets: dict[int, int] = {}
    position = len(header)
    max_id = font_id
    for obj_id in range(1, max_id + 1):
        rendered = f"{obj_id} 0 obj\n".encode("ascii") + objects[obj_id] + b"\nendobj\n"
        offsets[obj_id] = position
        chunks.append(rendered)
        position += len(rendered)

    xref_start = position
    xref = [f"xref\n0 {max_id + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for obj_id in range(1, max_id + 1):
        xref.append(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode("ascii")

    return b"".join(chunks) + b"".join(xref) + trailer
