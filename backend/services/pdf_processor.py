"""
PDF processing service for extracting and chunking text from documents
"""
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple
import re
import logging
import gc  # Added for memory management

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Target size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (full_text, page_text_dict)
        """
        try:
            full_text = ""
            page_texts = {}
            
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        page_texts[page_num] = text
                        full_text += f"\n{text}\n"
            
            if not full_text.strip():
                # Fallback to PyPDF2 if pdfplumber fails
                logger.warning(f"pdfplumber extraction empty, trying PyPDF2 for {pdf_path}")
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            page_texts[page_num + 1] = text
                            full_text += f"\n{text}\n"
            
            # Clean the text
            full_text = self._clean_text(full_text)
            page_texts = {k: self._clean_text(v) for k, v in page_texts.items()}
            
            logger.info(f"Extracted {len(full_text)} characters from {len(page_texts)} pages")
            return full_text, page_texts
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace which can mess up chunking math
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate provisional end position
            end = min(start + self.chunk_size, text_length)
            
            # If not at the very end of the document, look for a clean break
            if end < text_length:
                # Try to find a sentence boundary in the second half of the chunk
                sentence_end = text.rfind('.', start + self.chunk_size // 2, end)
                if sentence_end != -1:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    space_pos = text.rfind(' ', start, end)
                    
                    # CRITICAL FIX: Ensure picking this space doesn't cause a negative overlap loop
                    if space_pos > start + self.chunk_overlap:
                        end = space_pos
                    # If the space is too close to the start, ignore it and just hard-break at chunk_size
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_dict = {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                }
                
                if metadata:
                    chunk_dict.update(metadata)
                
                chunks.append(chunk_dict)
                chunk_id += 1
            
            # CRITICAL FIX: Prevent the infinite loop OOM crash
            next_start = end - self.chunk_overlap
            
            # If the calculated next start is somehow less than or equal to our current start,
            # we force it to move forward to the end of the current chunk so it doesn't get stuck.
            if next_start <= start:
                start = end
            else:
                start = next_start
        
        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")
        return chunks
    
    def process_pdf(self, pdf_path: str, document_id: str = None) -> List[Dict]:
        """
        Complete PDF processing pipeline: extract and chunk
        
        Args:
            pdf_path: Path to PDF file
            document_id: Optional document identifier
            
        Returns:
            List of processed chunks with metadata
        """
        filename = Path(pdf_path).name
        
        # Extract text
        full_text, page_texts = self.extract_text_from_pdf(pdf_path)
        
        if not full_text:
            raise ValueError(f"No text extracted from PDF: {filename}")
        
        # Create chunks with metadata
        metadata = {
            "source": filename,
            "document_id": document_id or filename,
            "total_pages": len(page_texts)
        }
        
        chunks = self.chunk_text(full_text, metadata)
        
        # Add page numbers to chunks (approximate based on character position)
        chars_per_page = len(full_text) / len(page_texts) if page_texts else len(full_text)
        for chunk in chunks:
            estimated_page = int(chunk["start_char"] / chars_per_page) + 1
            chunk["page"] = min(estimated_page, len(page_texts))
            
        # MEMORY OPTIMIZATION: Manually free up RAM used by the massive text strings
        del full_text
        del page_texts
        gc.collect()
        
        return chunks