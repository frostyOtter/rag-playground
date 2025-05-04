"""
PyMuPDF processor implementation for handling PDF documents.
Follows domain modeling approach with simple model first.
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
import pymupdf
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from loguru import logger

from src.data_service.data_base import DataProcessorBase
from src.models.common_pdf_models import (
    ChunkingStrategy,
    DocumentMetadata,
    DocumentChunk,
)


class PyMuPDFProcessor(DataProcessorBase):
    """
    PyMuPDF processor implementation for PDF documents.
    Analogous to a specialized PDF translator that converts documents between formats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize processor with optional configuration."""
        self.config = config or {}
        self.supported_format = ".pdf"  # Only PDF for now
        logger.info(f"Initialized PyMuPDFProcessor for PDF processing")

    def parse_document_to_markdown(
        self, input_path: str, output_path: str, **kwargs
    ) -> None:
        """
        Parse PDF document to markdown format.
        Think of this as translating a book from one language to another.
        """
        if not output_path.endswith(".md"):
            output_path += ".md"
        output_file = Path(output_path)

        logger.info(f"Converting {input_path} to markdown")

        try:
            for doc in self.read_documents(file_path=input_path):
                markdown_content = []

                # Process each page as a separate section
                for page_num in range(doc.page_count):
                    page = doc[page_num]

                    # Extract text with formatting hints
                    text = page.get_text("text")

                    # Simple markdown conversion
                    # In a more sophisticated approach, we'd analyze layout and formatting
                    markdown_content.append(f"# Page {page_num + 1}\n\n{text}\n\n")

                    logger.debug(f"Processed page {page_num + 1}/{doc.page_count}")

                # Join all pages and save
                full_markdown = "".join(markdown_content)
                output_file.write_text(full_markdown, encoding="utf-8")

                logger.success(f"Successfully converted to markdown: {output_path}")

        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            raise

    def parse_document_to_json(
        self, input_path: str, output_path: str, **kwargs
    ) -> None:
        """
        Parse PDF document to JSON format.
        Extracts structured data - like mining data from a document.
        """
        if not output_path.endswith(".json"):
            output_path += ".json"
        output_file = Path(output_path)

        logger.info(f"Converting {input_path} to JSON")

        try:
            for doc in self.read_documents(file_path=input_path):
                document_data = {"metadata": self._extract_metadata(doc), "pages": []}

                # Extract content from each page
                for page_num in range(doc.page_count):
                    page = doc[page_num]

                    page_data = {
                        "page_number": page_num + 1,
                        "text": page.get_text("text"),
                        "annotations": self._extract_annotations(page),
                        "links": page.get_links() if hasattr(page, "get_links") else [],
                        "dimensions": {
                            "width": page.rect.width,
                            "height": page.rect.height,
                        },
                    }

                    document_data["pages"].append(page_data)

                # Save to JSON
                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(document_data, f, indent=2, ensure_ascii=False)

                logger.success(f"Successfully converted to JSON: {output_path}")

        except Exception as e:
            logger.error(f"Error converting to JSON: {e}")
            raise

    def chunks_document(
        self,
        input_path: str,
        strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.FIXED_SIZE,
        chunk_size: int = 1000,
        overlap: int = 100,
        **kwargs,
    ) -> List[DocumentChunk]:
        """
        Chunk a PDF document into smaller pieces.
        Like dividing a cake into bite-sized pieces for easier consumption.
        """
        logger.info(f"Chunking document: {input_path}")

        # Convert string to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = ChunkingStrategy(strategy)
            except ValueError:
                logger.warning(
                    f"Unknown strategy '{strategy}', falling back to FIXED_SIZE"
                )
                strategy = ChunkingStrategy.FIXED_SIZE

        chunks = []

        try:
            for doc in self.read_documents(file_path=input_path):
                if strategy == ChunkingStrategy.PAGE_BASED:
                    chunks = self._chunk_by_page(doc)
                elif strategy == ChunkingStrategy.FIXED_SIZE:
                    chunks = self._chunk_by_size(doc, chunk_size, overlap)
                elif strategy == ChunkingStrategy.SEMANTIC:
                    # For now, use fixed size as fallback
                    logger.warning(
                        "Semantic chunking not implemented, using fixed size"
                    )
                    chunks = self._chunk_by_size(doc, chunk_size, overlap)

                # Add chunk IDs
                for idx, chunk in enumerate(chunks):
                    chunk.chunk_index = idx
                    chunk.chunk_id = f"chunk_{idx:04d}"

            logger.success(f"Successfully chunked document into {len(chunks)} pieces")

        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise

        return chunks

    def read_documents(
        self,
        file_path: Optional[str] = None,
        directory_path: Optional[str] = None,
        **kwargs,
    ) -> Iterator[pymupdf.Document]:
        """
        Read PDF documents from specified path.
        Acts as a PDF document reader.
        """
        if file_path:
            # Single file
            yield self._open_document(Path(file_path))
        elif directory_path:
            # Directory of files
            dir_path = Path(directory_path)
            if not dir_path.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")

            for file_path in dir_path.rglob("*"):
                if file_path.suffix.lower() == self.supported_format:
                    yield self._open_document(file_path)
        else:
            raise ValueError("Either file_path or directory_path must be provided")

    def _open_document(self, file_path: Path) -> pymupdf.Document:
        """Helper method to open a PDF document file."""
        if file_path.suffix.lower() != self.supported_format:
            raise ValueError(f"Only PDF files are supported. Got: {file_path.suffix}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.debug(f"Opening PDF document: {file_path}")
        return pymupdf.open(str(file_path))

    def _extract_metadata(self, doc: pymupdf.Document) -> Dict[str, Any]:
        """Extract document metadata - like reading the cover of a book."""
        metadata = doc.metadata or {}

        return {
            "format": metadata.get("format", "PDF"),
            "page_count": doc.page_count,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "creator": metadata.get("creator"),
            "producer": metadata.get("producer"),
            "creation_date": metadata.get("creationDate"),
            "modification_date": metadata.get("modDate"),
            "subject": metadata.get("subject"),
            "keywords": metadata.get("keywords"),
            "encryption": metadata.get("encryption"),
        }

    def _extract_annotations(self, page: pymupdf.Page) -> List[Dict[str, Any]]:
        """Extract annotations from a page - like finding bookmarks in a book."""
        annotations = []

        try:
            for annot in page.annots():
                if annot:
                    annotations.append(
                        {
                            "type": (
                                annot.type[1]
                                if isinstance(annot.type, tuple)
                                else annot.type
                            ),
                            "title": getattr(annot, "title", None),
                            "content": getattr(annot, "content", None),
                            "rect": (
                                list(annot.rect) if hasattr(annot, "rect") else None
                            ),
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not extract annotations: {e}")

        return annotations

    def _chunk_by_page(self, doc: pymupdf.Document) -> List[DocumentChunk]:
        """Chunk document by pages - like chapters in a book."""
        chunks = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")

            chunk = DocumentChunk(
                content=text,
                page_number=page_num + 1,
                metadata={"strategy": "page_based", "page_count": 1},
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_size(
        self, doc: pymupdf.Document, chunk_size: int, overlap: int
    ) -> List[DocumentChunk]:
        """Chunk document by fixed size - like cutting a scroll into equal pieces."""
        chunks = []
        current_chunk = ""
        current_page = 1
        chunk_metadata = {
            "strategy": "fixed_size",
            "chunk_size": chunk_size,
            "overlap": overlap,
        }

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")

            # Split text into words for more accurate chunking
            words = text.split()

            for word in words:
                if len(current_chunk) + len(word) + 1 <= chunk_size - overlap:
                    current_chunk += word + " "
                else:
                    # Save current chunk
                    chunks.append(
                        DocumentChunk(
                            content=current_chunk.strip(),
                            page_number=current_page,
                            metadata=chunk_metadata.copy(),
                        )
                    )

                    # Start new chunk with overlap
                    overlap_content = " ".join(current_chunk.split()[-overlap:])
                    current_chunk = overlap_content + " " + word + " "
                    current_page = page_num + 1

        # Add the last chunk if any content remains
        if current_chunk.strip():
            chunks.append(
                DocumentChunk(
                    content=current_chunk.strip(),
                    page_number=current_page,
                    metadata=chunk_metadata.copy(),
                )
            )

        return chunks


# Example usage
if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if len(input_path) == 0 or len(output_path) == 0:
        print("Please provide input and output paths.")
        sys.exit(1)

    # Determine if input is a directory or file
    if not input_path.endswith(".pdf"):
        # Input is a directory
        all_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".pdf")
        ]
        if len(all_files) == 0:
            print("No PDF files found in the directory.")
            sys.exit(1)

    else:
        # Input is a single PDF file
        all_files = [input_path]

    processor = PyMuPDFProcessor()

    for file in all_files:
        output_path = os.path.join(
            output_path, os.path.basename(file).replace(".pdf", ".md")
        )
        # Convert PDF to markdown
        processor.parse_document_to_markdown(file, output_path)

        # Convert PDF to JSON
        # processor.parse_document_to_json(input_path, output_path)

        # Chunk document
        # chunks = processor.chunks_document(file, strategy="page_based")

    # # Read documents from directory
    # for doc in processor.read_documents(directory_path="extracted_contexts"):
    #     # Process each document
    #     pass
