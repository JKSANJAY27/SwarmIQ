"""
SwarmIQ — Document Ingestion Pipeline
Processes PDF, MD, TXT, DOCX, HTML.
Falls back to Tesseract OCR for scanned PDFs.
"""

import logging
import os
import re

from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from ..config import Config

logger = logging.getLogger("swarmiq.ingestion")


class DocumentProcessor:
    """Reads various document formats into clean text."""

    @staticmethod
    def read_text(file_path: str) -> str:
        """Read a TXT or MD file."""
        import charset_normalizer
        
        with open(file_path, "rb") as f:
            raw = f.read()
        res = charset_normalizer.detect(raw)
        enc = res['encoding'] or 'utf-8'
        
        try:
            return raw.decode(enc)
        except Exception:
            return raw.decode('utf-8', errors='ignore')

    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Read PDF. Apply OCR if no text found."""
        try:
            doc = fitz.open(file_path)
            text_blocks = []
            
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_blocks.append(text)
                else:
                    # Try OCR if image-based
                    try:
                        import pytesseract
                        from PIL import Image
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        text_blocks.append(ocr_text)
                    except Exception as e:
                        logger.warning("OCR failed on page %d: %s", page.number, e)
                        
            return "\n\n".join(text_blocks)
        except Exception as e:
            logger.error("PyMuPDF failed on %s: %s", file_path, e)
            return ""

    @staticmethod
    def read_docx(file_path: str) -> str:
        """Read Word document."""
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            logger.error("python-docx not installed.")
            return ""

    @staticmethod
    def read_html(file_path: str) -> str:
        """Extract clean text from HTML file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator="\n")
        # collapse consecutive empty lines
        return re.sub(r'\n\s*\n', '\n\n', text).strip()

    @classmethod
    def process_file(cls, file_path: str) -> str:
        """Route file to correct processor based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        logger.info("Processing document: %s", file_path)
        
        if ext in (".txt", ".md", ".markdown", ".csv"):
            return cls.read_text(file_path)
        elif ext == ".pdf":
            return cls.read_pdf(file_path)
        elif ext == ".docx":
            return cls.read_docx(file_path)
        elif ext in (".html", ".htm"):
            return cls.read_html(file_path)
        else:
            logger.warning("Unsupported file type: %s", ext)
            return ""

    @classmethod
    def process_upload_batch(cls, file_paths: list[str]) -> str:
        """Process multiple files and concatenate with clear separators."""
        extracted = []
        for fp in file_paths:
            content = cls.process_file(fp)
            if content.strip():
                fname = os.path.basename(fp)
                extracted.append(f"--- START DOCUMENT: {fname} ---\n{content}\n--- END DOCUMENT: {fname} ---\n")
                
        return "\n".join(extracted)
