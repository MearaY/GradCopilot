"""
Paper Download Tool - Works with search_agent metadata structure.
"""
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Optional, Dict, Any
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ConfigDict
import arxiv
import re
import time

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_DOWNLOAD_DIR = Path("data")
DOWNLOAD_DELAY_SECONDS = 1.0


class PaperMetadata(BaseModel):
    """Paper metadata structure matching search_agent output."""
    model_config = ConfigDict(extra="allow")

    paper_id: str = Field(description="Paper ID, e.g., arXiv identifier")
    title: str = Field(description="Paper title")
    authors: List[str] = Field(description="List of authors")
    pdf_url: str = Field(description="Paper PDF download URL")
    published: str = Field(description="Publication year or date as string, e.g., '2025'")
    primary_category: str = Field(default="", description="Primary category, e.g., cs.LG")
    summary: str = Field(default="", description="Paper abstract")
    published_date: str = Field(default="", description="Publication date string, format: 2025-01-15T00:00:00Z")
    url: str = Field(default="", description="Paper detail page URL")
    categories: List[str] = Field(default_factory=list, description="List of categories")
    doi: Optional[str] = Field(default="", description="Paper DOI identifier")


class DownloadPapersInput(BaseModel):
    """Input model for paper download."""
    papers: List[PaperMetadata] = Field(
        description="List of paper metadata, format matches search_agent's paper_results. Each paper must include: paper_id, title, authors, pdf_url, published, primary_category."
    )
    target_dir: str = Field(
        default="",
        description="Target download directory. Defaults to 'data' if empty. Can be relative or absolute path, e.g., 'data/transformer', '/path/to/downloads'. Directory is created if it doesn't exist."
    )
    organize_by_category: bool = Field(
        default=False,
        description="Whether to organize files by category. If True, creates subdirectories like cs.LG, cs.AI. Default: False"
    )
    filename_format: str = Field(
        default="author_year_title",
        description="Filename format. Options: 'author_year_title' (default, e.g., 'Smith et al - 2023 - Transformer Architecture.pdf'), 'paper_id' (e.g., '2301.12345.pdf'), 'title' (title only). Default: 'author_year_title'"
    )


def _sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitize filename by removing illegal characters.
    
    Args:
        filename: Original filename
        max_length: Maximum length
        
    Returns:
        Sanitized filename
    """
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename


def _format_filename(paper: Dict[str, Any], format_type: str = "author_year_title") -> str:
    """
    Generate filename based on specified format.
    
    Args:
        paper: Paper metadata dictionary
        format_type: Format type
        
    Returns:
        Filename without extension
    """
    if format_type == "paper_id":
        return paper.get("paper_id", "unknown")
    
    elif format_type == "title":
        title = paper.get("title", "unknown")
        return _sanitize_filename(title)
    
    elif format_type == "author_year_title":
        authors = paper.get("authors", [])
        if authors:
            if len(authors) == 1:
                author_str = authors[0].split()[-1]
            else:
                author_str = authors[0].split()[-1] + " et al"
        else:
            author_str = "Unknown"
        
        year = paper.get("published", "Unknown")
        title = paper.get("title", "Unknown")
        title = _sanitize_filename(title, max_length=150)
        
        return f"{author_str} - {year} - {title}"
    
    else:
        return paper.get("paper_id", "unknown")


def _get_download_path(paper: Dict[str, Any], target_dir: Path, organize_by_category: bool, filename_format: str) -> Path:
    """
    Get download path for a paper.
    
    Args:
        paper: Paper metadata dictionary
        target_dir: Target directory
        organize_by_category: Whether to organize by category
        filename_format: Filename format
        
    Returns:
        Full file path
    """
    if organize_by_category:
        category = paper.get("primary_category", "uncategorized")
        category = category.replace(".", "_") if category else "uncategorized"
        download_dir = target_dir / category
    else:
        download_dir = target_dir
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    filename = _format_filename(paper, filename_format)
    filepath = download_dir / f"{filename}.pdf"
    
    return filepath


def _download_pdf_from_url(pdf_url: str, filepath: Path, max_retries: int = 3) -> bool:
    """
    Download PDF file from URL.
    
    Args:
        pdf_url: PDF URL
        filepath: Save path
        max_retries: Maximum retry attempts
        
    Returns:
        Whether download succeeded
    """
    if filepath.exists():
        logger.info(f"File already exists, skipping: {filepath}")
        return True
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading: {pdf_url} -> {filepath} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Download successful: {filepath} ({filepath.stat().st_size / 1024:.2f} KB)")
            return True
            
        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Download failed after {max_retries} attempts: {pdf_url}")
                return False
    
    return False


def _download_paper_using_arxiv(paper_id: str, filepath: Path) -> bool:
    """
    Download paper using arxiv library (fallback method).
    
    Args:
        paper_id: Paper ID
        filepath: Save path
        
    Returns:
        Whether download succeeded
    """
    if filepath.exists():
        logger.info(f"File already exists, skipping: {filepath}")
        return True
    
    try:
        logger.info(f"Downloading using arxiv library: {paper_id} -> {filepath}")
        
        search = arxiv.Search(id_list=[paper_id])
        client = arxiv.Client()
        results = list(client.results(search))
        
        if not results:
            logger.error(f"Paper not found: {paper_id}")
            return False
        
        result = results[0]
        result.download_pdf(dirpath=str(filepath.parent), filename=filepath.name)
        
        logger.info(f"Download successful: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Download using arxiv library failed: {str(e)}")
        return False


async def _download_papers(
    papers: List[Dict[str, Any]],
    target_dir: Optional[str] = None,
    organize_by_category: bool = False,
    filename_format: str = "author_year_title"
) -> Dict[str, Any]:
    """
    Internal implementation for downloading papers.
    
    Args:
        papers: List of paper metadata
        target_dir: Target download directory
        organize_by_category: Whether to organize by category
        filename_format: Filename format
        
    Returns:
        Download statistics
    """
    if not papers:
        logger.warning("Paper list is empty, nothing to download")
        return {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "downloaded_files": [],
            "failed_papers": []
        }
    
    if target_dir:
        download_base = Path(target_dir)
    else:
        download_base = DEFAULT_DOWNLOAD_DIR
    
    download_base = download_base.resolve()
    download_base.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download of {len(papers)} papers to directory: {download_base}")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    downloaded_files = []
    failed_papers = []
    
    for i, paper in enumerate(papers, 1):
        try:
            paper_id = paper.get("paper_id", "unknown")
            pdf_url = paper.get("pdf_url")
            title = paper.get("title", "Unknown")
            
            logger.info(f"[{i}/{len(papers)}] Processing paper: {title} ({paper_id})")
            
            filepath = _get_download_path(paper, download_base, organize_by_category, filename_format)
            
            if filepath.exists():
                logger.info(f"File already exists, skipping: {filepath}")
                skipped_count += 1
                downloaded_files.append(str(filepath))
                continue
            
            success = False
            if pdf_url:
                success = _download_pdf_from_url(pdf_url, filepath)
            
            if not success and paper_id and paper_id != "unknown":
                success = _download_paper_using_arxiv(paper_id, filepath)
            
            if success:
                success_count += 1
                downloaded_files.append(str(filepath))
            else:
                failed_count += 1
                failed_papers.append({
                    "paper_id": paper_id,
                    "title": title,
                    "reason": "Download failed"
                })
            
            if i < len(papers):
                time.sleep(DOWNLOAD_DELAY_SECONDS)
                
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            failed_count += 1
            failed_papers.append({
                "paper_id": paper.get("paper_id", "unknown"),
                "title": paper.get("title", "Unknown"),
                "reason": str(e)
            })
    
    result = {
        "success": success_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "total": len(papers),
        "downloaded_files": downloaded_files,
        "failed_papers": failed_papers
    }
    
    logger.info(f"Download completed: success {success_count}, failed {failed_count}, skipped {skipped_count}, total {len(papers)}")
    return result


async def download_papers(
    papers: List[PaperMetadata],
    target_dir: Optional[str] = None,
    organize_by_category: bool = False,
    filename_format: str = "author_year_title"
) -> Dict[str, Any]:
    """
    Download paper PDF files to local directory.
    
    Args:
        papers: List of paper metadata
        target_dir: Target download directory (optional)
        organize_by_category: Whether to organize by category (default: False)
        filename_format: Filename format (default: 'author_year_title')
        
    Returns:
        Download statistics including success, failed, skipped counts and file paths
    """
    try:
        logger.info(f"Calling download tool: papers={len(papers)}, target_dir={target_dir}, organize_by_category={organize_by_category}")
        
        parsed_papers = [
            paper.model_dump(exclude_none=True)
            if isinstance(paper, PaperMetadata)
            else paper
            for paper in papers
        ]

        result = await _download_papers(
            papers=parsed_papers,
            target_dir=target_dir,
            organize_by_category=organize_by_category,
            filename_format=filename_format
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error downloading papers: {str(e)}")
        raise Exception(f"Error downloading papers: {str(e)}")


def generate_mock_paper_data(
    paper_id: str = "2511.02593",
    title: str = "A Large Language Model for Corporate Credit Scoring",
    authors: Optional[List[str]] = None,
    published: str = "2025",
    primary_category: str = "cs.LG"
) -> Dict[str, Any]:
    """
    Generate mock paper data for testing.
    
    Args:
        paper_id: Paper ID
        title: Paper title
        authors: List of authors
        published: Publication year
        primary_category: Primary category
        
    Returns:
        Mock paper metadata dictionary
    """
    if authors is None:
        authors = [
            "Chitro Majumdar",
            "Sergio Scandizzo",
            "Ratanlal Mahanta",
            "Avradip Mandal",
            "Swarnendu Bhattacharjee"
        ]
    
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "summary": (
            "This paper presents a large language model approach for corporate credit "
            "scoring, addressing the challenges in financial risk assessment using advanced NLP techniques."
        ),
        "published": str(published),
        "published_date": f"{published}-01-15T00:00:00Z",
        "url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
        "primary_category": primary_category,
        "categories": [primary_category],
        "doi": f"10.48550/arXiv.{paper_id}"
    }


def generate_mock_papers_list(count: int = 1) -> List[Dict[str, Any]]:
    """
    Generate multiple mock paper data for testing.
    
    Args:
        count: Number of papers to generate (default: 1)
        
    Returns:
        List of mock papers
    """
    mock_papers = []
    
    sample_paper = {
        "paper_id": "2511.02593",
        "title": "A Large Language Model for Corporate Credit Scoring",
        "authors": [
            "Chitro Majumdar",
            "Sergio Scandizzo",
            "Ratanlal Mahanta",
            "Avradip Mandal",
            "Swarnendu Bhattacharjee"
        ],
        "published": "2025",
        "primary_category": "cs.LG"
    }
    
    mock_papers.append(generate_mock_paper_data(**sample_paper))
    
    for i in range(1, count):
        mock_papers.append(generate_mock_paper_data(
            paper_id=f"2511.{10000+i:05d}",
            title=f"Sample Paper {i+1} on AI Research",
            authors=[f"Author {i+1}", f"Co-author {i+1}"],
            published="2025",
            primary_category="cs.LG"
        ))
    
    return mock_papers


async def test_download_with_mock_data():
    """Test download function with mock data."""
    print("=" * 70)
    print("Testing Paper Download Tool (with mock data)")
    print("=" * 70)
    
    mock_papers = generate_mock_papers_list(count=1)
    
    print(f"\nGenerated {len(mock_papers)} mock papers:")
    for i, paper in enumerate(mock_papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   ID: {paper['paper_id']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   PDF URL: {paper['pdf_url']}")
    
    print("\n" + "=" * 70)
    print("Starting download test...")
    print("=" * 70)
    
    result = await _download_papers(
        papers=mock_papers,
        target_dir="data/test_downloads",
        organize_by_category=True,
        filename_format="author_year_title"
    )
    
    print(f"\nDownload results:")
    print(f"  Success: {result['success']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Skipped: {result['skipped']}")
    print(f"  Total: {result['total']}")
    
    if result['downloaded_files']:
        print(f"\nDownloaded files:")
        for filepath in result['downloaded_files']:
            print(f"  - {filepath}")
    
    if result['failed_papers']:
        print(f"\nFailed papers:")
        for paper in result['failed_papers']:
            print(f"  - {paper['title']}: {paper['reason']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_download_with_mock_data())
