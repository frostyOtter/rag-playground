"""
Common data models for PDF processing.
Following domain modeling principles with simple, focused models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class ChunkingStrategy(Enum):
    """PDF document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    PAGE_BASED = "page_based"


@dataclass
class DocumentMetadata:
    """PDF document metadata model - simple structure to start with."""

    format: str
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    encryption: Optional[str] = None
    additional_metadata: Dict[str, Any] = None


@dataclass
class DocumentChunk:
    """PDF document chunk model - basic building block for processing."""

    content: str
    page_number: int
    metadata: Dict[str, Any]
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None


# @dataclass
# class PageAnnotation:
#     """PDF page annotation model."""

#     type: str
#     title: Optional[str] = None
#     content: Optional[str] = None
#     rect: Optional[List[float]] = None


# @dataclass
# class PageLink:
#     """PDF page link model."""

#     kind: int  # Link type
#     from_rect: List[float]  # Source rectangle
#     to_page: Optional[int] = None  # Target page
#     uri: Optional[str] = None  # External URI
#     title: Optional[str] = None  # Link title
