"""
PDF Parsing Tool - Extract paper content for vectorization.
"""
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
import re

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class ParsePDFInput(BaseModel):
    """Input model for PDF parsing."""
    pdf_path: str = Field(
        description="Path to PDF file. Example: 'data/example.pdf' or '/path/to/paper.pdf'"
    )
    chunk_size: int = Field(
        default=1000,
        description="Text chunk size in characters. Default: 1000"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in characters. Default: 200"
    )
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract PDF metadata. Default: True"
    )


def _clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    return text.strip()


def _split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks based on paragraph and sentence boundaries.
    
    Args:
        text: Text to split
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap size in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            para_end = text.rfind('\n\n', start, end)
            if para_end != -1 and (end - para_end) < chunk_size * 0.3:
                end = para_end + 2
            else:
                sent_end = text.rfind('. ', start, end)
                if sent_end != -1 and (end - sent_end) < chunk_size * 0.3:
                    end = sent_end + 2
                elif sent_end == -1:
                    for punc in ['! ', '? ', '; ', '。', '！', '？', '；']:
                        punc_end = text.rfind(punc, start, end)
                        if punc_end != -1 and (end - punc_end) < chunk_size * 0.3:
                            end = punc_end + len(punc)
                            break
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        start = end - chunk_overlap
        
        if start >= len(text):
            break
    
    return chunks


def _extract_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract PDF metadata.
    
    Args:
        pdf_path: PDF file path
        
    Returns:
        Metadata dictionary
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "file_path": str(pdf_path),
            "file_name": pdf_path.name
        }
    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {str(e)}")
        return {
            "file_path": str(pdf_path),
            "file_name": pdf_path.name
        }


async def _parse_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    extract_metadata: bool = True
) -> Dict[str, Any]:
    """
    Internal implementation for PDF parsing.
    
    Args:
        pdf_path: PDF file path
        chunk_size: Text chunk size
        chunk_overlap: Text chunk overlap
        extract_metadata: Whether to extract metadata
        
    Returns:
        Dictionary containing chunks and metadata
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"File is not PDF format: {pdf_path}")
    
    logger.info(f"Starting PDF parsing: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text += text + "\n"
        
        doc.close()
        
        full_text = _clean_text(full_text)
        
        if not full_text.strip():
            logger.warning(f"PDF has no extractable text: {pdf_path}")
            return {
                "chunks": [],
                "total_chunks": 0,
                "total_chars": 0,
                "metadata": {}
            }
        
        chunks = _split_text(full_text, chunk_size, chunk_overlap)
        
        metadata = {}
        if extract_metadata:
            metadata = _extract_pdf_metadata(pdf_file)
        
        result = {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "total_chars": len(full_text),
            "metadata": metadata
        }
        
        logger.info(f"PDF parsing completed: {len(chunks)} chunks, {len(full_text)} chars")
        
        return result
        
    except Exception as e:
        logger.error(f"PDF parsing failed: {str(e)}")
        raise Exception(f"PDF parsing failed: {str(e)}")


async def parse_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    extract_metadata: bool = True
) -> Dict[str, Any]:
    """
    Parse PDF file, extract text content and split into chunks for vectorization and RAG.
    
    Args:
        pdf_path: PDF file path
        chunk_size: Text chunk size (default: 1000)
        chunk_overlap: Text chunk overlap (default: 200)
        extract_metadata: Whether to extract metadata (default: True)
        
    Returns:
        Dictionary containing chunks, total_chunks, total_chars, and metadata
    """
    try:
        logger.info(f"Calling PDF parser: pdf_path={pdf_path}, chunk_size={chunk_size}")
        
        result = await _parse_pdf(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_metadata=extract_metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise Exception(f"Error parsing PDF: {str(e)}")
