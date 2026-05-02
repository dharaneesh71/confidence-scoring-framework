"""
PDF Processor — Text extraction and chunking for ChromaDB ingestion.

ROOT CAUSE FIX:
    pdfplumber silently returns empty pages for XSL-Formatter / Antenna House
    generated PDFs (like the NLTK O'Reilly book). This version uses
    pdftotext (poppler) as the primary extractor with pypdf as a fallback —
    both reliably extract the 1M+ characters that pdfplumber misses entirely.

INSTALL POPPLER (one-time):
    macOS  : brew install poppler
    Ubuntu : sudo apt-get install poppler-utils
    Docker : apt-get install -y poppler-utils  (in your Dockerfile)
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Chunking parameters ────────────────────────────────────────────────────────
CHUNK_SIZE    = 700    # characters per chunk (~115 words)
CHUNK_OVERLAP = 140    # 20% overlap to preserve cross-boundary context
MIN_CHUNK_LEN = 80     # drop fragments (page numbers, headers, whitespace runs)


class PDFProcessor:
    """
    Extracts text from PDFs and splits into overlapping chunks ready for
    ChromaService.add_documents().
    """

    # ── Text extraction ────────────────────────────────────────────────────────

    def _extract_via_pdftotext(self, pdf_path: str) -> str:
        """
        Primary extraction: pdftotext -layout preserves reading order across
        multi-column layouts and handles XSL Formatter PDFs correctly.
        Returns empty string on failure so the fallback can run.
        """
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info(
                    "[PDFProcessor] pdftotext: %d chars from '%s'",
                    len(result.stdout),
                    Path(pdf_path).name,
                )
                return result.stdout
            if result.returncode != 0:
                logger.warning(
                    "[PDFProcessor] pdftotext exit %d: %s",
                    result.returncode,
                    result.stderr[:200],
                )
        except FileNotFoundError:
            logger.warning(
                "[PDFProcessor] pdftotext not found. "
                "Install poppler: brew install poppler  OR  apt-get install poppler-utils. "
                "Falling back to pypdf."
            )
        except subprocess.TimeoutExpired:
            logger.warning("[PDFProcessor] pdftotext timed out. Falling back to pypdf.")
        except Exception as exc:
            logger.warning("[PDFProcessor] pdftotext error: %s. Falling back.", exc)
        return ""

    def _extract_via_pypdf(self, pdf_path: str) -> str:
        """Fallback extraction using pypdf (pip install pypdf)."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            pages  = [page.extract_text() or "" for page in reader.pages]
            text   = "\n".join(pages)
            logger.info(
                "[PDFProcessor] pypdf: %d chars from '%s'",
                len(text),
                Path(pdf_path).name,
            )
            return text
        except Exception as exc:
            logger.error("[PDFProcessor] pypdf also failed: %s", exc)
            return ""

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF.
        Tries pdftotext first, pypdf second.
        Raises ValueError if both fail (e.g. scanned image PDF).
        """
        text = self._extract_via_pdftotext(pdf_path)
        if not text.strip():
            text = self._extract_via_pypdf(pdf_path)
        if not text.strip():
            raise ValueError(
                f"Could not extract text from '{Path(pdf_path).name}'. "
                "The file may be a scanned image PDF — OCR is required for those."
            )
        return text

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int    = CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Sliding-window chunker that extends each window to the nearest
        sentence boundary so the LLM never receives a half-sentence.

        Steps:
          1. Move a window of `chunk_size` chars through the text.
          2. Look ahead up to 250 chars for a sentence-ending punctuation.
          3. Extend the window end to that boundary.
          4. Advance start by (chunk_size - overlap) chars so windows overlap.
          5. Drop any chunk shorter than MIN_CHUNK_LEN.
        """
        chunks: List[str] = []
        step  = max(chunk_size - overlap, 50)
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                look_ahead = text[end: end + 250]
                for punct in (".\n", "!\n", "?\n", ". ", "! ", "? "):
                    idx = look_ahead.find(punct)
                    if idx != -1:
                        end = end + idx + len(punct)
                        break

            chunk = text[start:end].strip()
            if len(chunk) >= MIN_CHUNK_LEN:
                chunks.append(chunk)

            start += step

        logger.info(
            "[PDFProcessor] %d chunks from %d chars (avg %d chars/chunk)",
            len(chunks),
            len(text),
            int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0,
        )
        return chunks

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_pdf(
        self,
        pdf_path: str,
        document_id: Optional[str] = None,
        chunk_size: int = CHUNK_SIZE,
        overlap: int    = CHUNK_OVERLAP,
    ) -> List[Dict]:
        """
        Full pipeline: extract text → chunk → return chunk dicts for ChromaDB.

        Each returned dict has:
            text        : chunk text content
            document_id : source filename (for ChromaDB ID generation)
            source      : same as document_id (used for domain filter lookup)
            chunk_id    : sequential integer index
            page        : 0 (page tracking not available for this PDF type)
            char_count  : character length of the chunk

        Usage:
            processor = PDFProcessor()
            chunks = processor.process_pdf("data/uploads/NLTK.pdf", "NLTK.pdf")
            chroma_service.add_documents(chunks, domain="NLP")
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = document_id or path.name
        logger.info("[PDFProcessor] Processing '%s' (doc_id=%s)", path.name, doc_id)

        raw_text = self.extract_text(pdf_path)
        if not raw_text.strip():
            logger.error("[PDFProcessor] No text extracted from '%s'", path.name)
            return []

        chunks = self._split_into_chunks(raw_text, chunk_size, overlap)
        if not chunks:
            logger.error("[PDFProcessor] Chunking produced 0 results for '%s'", path.name)
            return []

        result = [
            {
                "text":        chunk,
                "document_id": doc_id,
                "source":      doc_id,
                "chunk_id":    i,
                "page":        0,
                "char_count":  len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]

        logger.info(
            "[PDFProcessor] '%s' → %d chunks ready for ChromaDB",
            path.name,
            len(result),
        )
        return result