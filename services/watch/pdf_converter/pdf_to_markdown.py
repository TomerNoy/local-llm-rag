#!/usr/bin/env python3
"""
PDF to Markdown converter for RAG pipeline ingestion.
Handles: English, Hebrew, mixed, tables, scanned PDFs.

Usage:
    python pdf_to_markdown.py <pdf_file> [output.md]

As a library:
    from pdf_to_markdown import pdf_to_markdown
    markdown = pdf_to_markdown("document.pdf")
"""

import sys
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _is_scanned(pdf_path):
    """Check if PDF is scanned (image-based, no real text layer)."""
    import fitz

    doc = fitz.open(pdf_path)
    text_chars = 0
    image_count = 0
    pages = min(len(doc), 3)

    for i in range(pages):
        page = doc[i]
        text_chars += len(page.get_text().strip())
        image_count += len(page.get_images())

    doc.close()
    return image_count > 0 and text_chars < 50 * pages


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_with_pymupdf4llm(pdf_path):
    """Structured markdown via pymupdf4llm (preserves tables & headers).
    Returns None if unavailable or output quality is too low."""
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(str(pdf_path), write_images=False)

        # If too many unmappable chars, this PDF has encoding issues —
        # the basic extractor handles those better.
        if len(md) > 0 and md.count("\ufffd") / len(md) > 0.02:
            log.info("pymupdf4llm output has encoding issues, falling back")
            return None
        return md
    except ImportError:
        return None
    except Exception as e:
        log.warning("pymupdf4llm failed: %s", e)
        return None


def _extract_with_pymupdf_basic(pdf_path):
    """Plain block-level text extraction via PyMuPDF. Reliable fallback."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1] / 10), -b[0]))
        lines = [b[4].strip() for b in blocks if b[4].strip()]
        if lines:
            pages.append("\n\n".join(lines))

    doc.close()
    return "\n\n---\n\n".join(pages)


def _extract_scanned(pdf_path):
    """OCR extraction for scanned PDFs via Docling + Tesseract."""
    import os
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if "TESSDATA_PREFIX" not in os.environ:
        for p in ["/opt/homebrew/share/tessdata/",
                   "/usr/local/share/tessdata/",
                   "/usr/share/tesseract-ocr/4.00/tessdata/"]:
            if os.path.exists(p):
                os.environ["TESSDATA_PREFIX"] = p
                break

    ocr_options = TesseractOcrOptions()
    ocr_options.lang = ["heb", "eng"]
    ocr_options.force_full_page_ocr = True

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Common Latin-Extended → Hebrew character corruption from PDFs with
# non-standard font encodings.  Safe to apply globally — these Latin
# Extended chars virtually never appear in real mixed-language content.
_HEBREW_CHAR_FIXES = {
    "Ě": "מ", "ĝ": "ס", "Ĥ": "ר", "Ĕ": "ט", "ĕ": "י", "ė": "כ", "ĥ": "ש",
    "Ď": "ג", "Ğ": "ע", "ď": "ד", "Ē": "ז", "ę": "ם", "ě": "ן", "Ė": "ת",
    "ğ": "נ", "Ĝ": "צ", "Ĩ": "ח", "ħ": "ל", "Ġ": "פ",
    "ġ": "ף", "Ģ": "ץ", "ģ": "ק", "ĵ": "א", "Ķ": "ב",
    "ķ": "ו", "ĸ": "ה", "Ĺ": "נ", "ĺ": "ן", "Ļ": "ם",
}


def _fix_hebrew_encoding(text):
    """Replace Latin-Extended chars that are actually mis-encoded Hebrew."""
    for bad, good in _HEBREW_CHAR_FIXES.items():
        text = text.replace(bad, good)
    return text


def _fix_reversed_table_cells(text):
    """Fix reversed Hebrew text in markdown table cells.
    Some PDF extractors output table cells in visual (RTL display) order
    instead of logical order. Uses word-frequency scoring to detect this."""
    try:
        from wordfreq import word_frequency
    except ImportError:
        return text  # can't detect without wordfreq — leave as-is

    def is_reversed(s):
        words = s.split()
        if not words:
            return False
        fwd = sum(word_frequency(w, "he") for w in words)
        rev = sum(word_frequency(w[::-1], "he") for w in words)
        return rev > fwd

    def reverse_segment(s):
        r = s[::-1]
        r = re.sub(r'[a-zA-Z0-9@/\-\.,:_]+', lambda m: m.group(0)[::-1], r)
        r = r.translate(str.maketrans("()[]{}", ")(][}{"))
        return r.strip()

    hebrew_re = re.compile(r'^[\u0590-\u05FF\s"\'.:,;!?()\-]+$')
    lines = text.split("\n")
    out = []

    for line in lines:
        if line.startswith("|") and not re.match(r"^\|[-:\s|]+\|$", line):
            cells = line.split("|")
            fixed = []
            for cell in cells:
                s = cell.strip()
                if s and hebrew_re.match(s) and is_reversed(s):
                    fixed.append(" " + reverse_segment(s) + " ")
                else:
                    fixed.append(cell)
            out.append("|".join(fixed))
        else:
            out.append(line)

    return "\n".join(out)


def _clean_for_rag(text):
    """General cleanup for RAG / vector DB ingestion."""
    # Drop image references — not useful for text retrieval
    text = re.sub(r"!\[.*?\]\(.*?\)\s*", "", text)
    text = re.sub(r"<!--\s*image\s*-->\s*", "", text)

    # Drop replacement characters
    text = text.replace("\ufffd", "")

    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def pdf_to_markdown(pdf_path, output_path=None):
    """Convert a PDF to markdown ready for RAG ingestion.

    Args:
        pdf_path: Path to the PDF file.
        output_path: Where to save .md. Defaults to <pdf_name>.md.

    Returns:
        Markdown string, or None on failure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        log.error("File not found: %s", pdf_path)
        return None

    log.info("Processing: %s", pdf_path.name)

    # 1. Extract
    try:
        if _is_scanned(pdf_path):
            log.info("Scanned PDF → OCR extraction")
            text = _extract_scanned(pdf_path)
        else:
            log.info("Digital PDF → structured extraction")
            text = _extract_with_pymupdf4llm(pdf_path)
            if not text:
                log.info("Falling back to basic extraction")
                text = _extract_with_pymupdf_basic(pdf_path)
    except Exception as e:
        log.error("Extraction failed: %s", e)
        return None

    if not text or not text.strip():
        log.error("No text extracted")
        return None

    # 2. Post-process
    text = _fix_hebrew_encoding(text)
    text = _fix_reversed_table_cells(text)
    text = _clean_for_rag(text)

    # 3. Save
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    Path(output_path).write_text(text, encoding="utf-8")
    log.info("Saved: %s (%d chars)", output_path, len(text))

    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage: python pdf_to_markdown.py <pdf_file> [output.md]")
        sys.exit(1)

    result = pdf_to_markdown(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
    sys.exit(0 if result else 1)
