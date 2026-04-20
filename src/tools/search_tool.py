"""
arXiv Paper Search Tool
"""
from typing import List, Optional, Dict, Any, Union
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.utils.log_utils import setup_logger
import arxiv
from datetime import datetime

logger = setup_logger(__name__)


class ArxivSearchInput(BaseModel):
    """Input model for arXiv search."""
    querys: List[str] = Field(
        description="List of search keywords in English. Example: ['transformer', 'machine translation']"
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of results to return, default 50"
    )
    sort_by: str = Field(
        default="Relevance",
        description="Sort method: 'Relevance', 'LastUpdatedDate', 'SubmittedDate'"
    )
    sort_order: str = Field(
        default="Descending",
        description="Sort order: 'Ascending', 'Descending'"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date, format: YYYY-MM-DD, example: '2023-01-01'"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date, format: YYYY-MM-DD, example: '2024-12-31'"
    )


def _format_date(date: Union[str, datetime]) -> str:
    """
    Format date for arXiv API.

    Args:
        date: Date string or datetime object

    Returns:
        Formatted date string in YYYYMMDD0000 format
    """
    if isinstance(date, datetime):
        return date.strftime("%Y%m%d0000")
    elif isinstance(date, str):
        if len(date) == 4 and date.isdigit():
            year = int(date)
            if year == datetime.now().year:
                return datetime.now().strftime("%Y%m%d0000")
            else:
                parsed_date = datetime(year, 1, 1)
                return parsed_date.strftime("%Y%m%d0000")

        date_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%Y-%m",
            "%Y/%m",
        ]

        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date, fmt)
                return parsed_date.strftime("%Y%m%d0000")
            except ValueError:
                continue

        try:
            from dateutil import parser
            parsed_date = parser.parse(date)
            return parsed_date.strftime("%Y%m%d0000")
        except:
            return datetime.now().strftime("%Y%m%d0000")

    return datetime.now().strftime("%Y%m%d0000")


def _parse_paper_result(result: arxiv.Result) -> Dict[str, Any]:
    """
    Parse arXiv search result to dictionary.

    Args:
        result: arXiv Result object

    Returns:
        Parsed paper dictionary
    """
    paper_id = result.get_short_id()
    published_year = result.published.year if result.published else None

    return {
        "paper_id": paper_id,
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "summary": result.summary,
        "published": published_year,
        "published_date": result.published.isoformat() if result.published else None,
        "url": result.entry_id,
        "pdf_url": result.pdf_url,
        "primary_category": result.primary_category,
        "categories": result.categories,
        "doi": result.doi if hasattr(result, 'doi') else None
    }


def _format_papers_list(search_results) -> List[Dict[str, Any]]:
    """
    Format search results list.

    Args:
        search_results: arXiv search results

    Returns:
        List of formatted paper dictionaries
    """
    results_list = list(search_results)
    formatted_papers = [_parse_paper_result(result) for result in results_list]
    logger.info(f"Formatting paper list, total {len(results_list)} papers")
    return formatted_papers


async def _search_papers(
    querys: List[str],
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> List[Dict[str, Any]]:
    """
    Internal search implementation.

    Args:
        querys: List of search keywords
        max_results: Maximum number of results
        sort_by: Sort criterion
        sort_order: Sort order
        start_date: Start date filter
        end_date: End date filter

    Returns:
        List of found papers
    """
    try:
        if querys:
            agent_parts = [f'all:"{term}"' for term in querys]
            search_query = " OR ".join(agent_parts)
            search_query = f"({search_query})"
        else:
            search_query = ""

        logger.info(f"Starting paper search: query='{search_query}'")

        try:
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order
            )
        except Exception as e:
            logger.error(f"Failed to create arXiv search: {str(e)}")
            return []

        results_list = list(search.results())
        logger.info(f"Raw search results: {len(results_list)} papers")

        has_valid_date = False
        if start_date and start_date.strip() != "":
            has_valid_date = True
        if end_date and end_date.strip() != "":
            has_valid_date = True

        if has_valid_date:
            filtered_results = []
            for result in results_list:
                if result.published:
                    year = result.published.year
                    start_year = None
                    if start_date and start_date.strip() != "":
                        if isinstance(start_date, str) and len(start_date) == 4 and start_date.isdigit():
                            start_year = int(start_date)
                        else:
                            try:
                                start_year = datetime.strptime(_format_date(start_date), "%Y%m%d0000").year
                            except:
                                start_year = None

                    end_year = None
                    if end_date and end_date.strip() != "":
                        if isinstance(end_date, str) and len(end_date) == 4 and end_date.isdigit():
                            end_year = int(end_date)
                        else:
                            try:
                                end_year = datetime.strptime(_format_date(end_date), "%Y%m%d0000").year
                            except:
                                end_year = None

                    in_range = True
                    if start_year is not None and year < start_year:
                        in_range = False
                    if end_year is not None and year > end_year:
                        in_range = False

                    if in_range:
                        filtered_results.append(result)

            logger.info(f"After date filtering: {len(filtered_results)} papers")
            results_list = filtered_results
        else:
            logger.info(f"No date filter set, returning all results")

        papers = _format_papers_list(results_list)
        logger.info(f"Paper search completed, found {len(papers)} papers")
        return papers
    except Exception as e:
        logger.error(f"Paper search failed: {str(e)}")
        raise


async def arxiv_search_papers(
    querys: List[str],
    max_results: int = 50,
    sort_by: str = "Relevance",
    sort_order: str = "Descending",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search arXiv papers.

    Supports keyword, date, and sorting filters.

    Args:
        querys: List of search keywords
        max_results: Maximum number of results
        sort_by: Sort method ('Relevance', 'LastUpdatedDate', 'SubmittedDate')
        sort_order: Sort order ('Ascending', 'Descending')
        start_date: Start date filter
        end_date: End date filter

    Returns:
        List of found papers
    """
    try:
        logger.info(f"Calling arXiv search: querys={querys}")

        sort_criterion_map = {
            "Relevance": arxiv.SortCriterion.Relevance,
            "LastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "SubmittedDate": arxiv.SortCriterion.SubmittedDate
        }
        sort_order_map = {
            "Ascending": arxiv.SortOrder.Ascending,
            "Descending": arxiv.SortOrder.Descending
        }

        sort_by_criterion = sort_criterion_map.get(sort_by, arxiv.SortCriterion.Relevance)
        sort_order_enum = sort_order_map.get(sort_order, arxiv.SortOrder.Descending)

        results = await _search_papers(
            querys=querys,
            max_results=max_results,
            sort_by=sort_by_criterion,
            sort_order=sort_order_enum,
            start_date=start_date,
            end_date=end_date
        )

        return results

    except Exception as e:
        logger.error(f"arXiv search failed: {str(e)}")
        raise Exception(f"Error searching papers: {str(e)}")
