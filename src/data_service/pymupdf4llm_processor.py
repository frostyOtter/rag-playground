"""
PyMuPDF4LLM processor implementation for handling PDF documents.
This processor leverages PyMuPDF4LLM for advanced markdown conversion and LLM optimization.
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
import pymupdf4llm
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from loguru import logger
from langchain.text_splitter import MarkdownTextSplitter

from src.data_service.data_base import DataProcessorBase
from src.models.common_pdf_models import (
    ChunkingStrategy,
    # DocumentMetadata,
    DocumentChunk,
    # PageAnnotation,
    # PageLink,
)
from src.data_service.pymupdf_processor import PyMuPDFProcessor


class PyMuPDF4LLMProcessor(DataProcessorBase):
    """
    PyMuPDF4LLM processor implementation for PDF documents.
    Analogous to a specialized document translator optimized for LLM consumption.

    This processor is particularly useful for RAG (Retrieval-Augmented Generation) pipelines,
    providing enhanced markdown output with proper semantic structure for LLMs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize processor with optional configuration."""
        self.config = config or {}
        self.supported_format = ".pdf"
        self.pymupdf_processor = PyMuPDFProcessor(config)
        logger.info(
            f"Initialized PyMuPDF4LLMProcessor for LLM-optimized PDF processing"
        )

    def parse_document_to_markdown(
        self, input_path: str, output_path: str, **kwargs
    ) -> None:
        """
        Parse PDF document to LLM-optimized markdown format.
        Think of this as creating a script that an LLM can easily understand and perform from.

        Uses PyMuPDF4LLM for enhanced markdown conversion with table preservation.
        """
        output_file = Path(output_path)

        logger.info(f"Converting {input_path} to LLM-optimized markdown")

        try:
            # Use PyMuPDF4LLM for enhanced markdown conversion
            md_text = pymupdf4llm.to_markdown(
                input_path, **kwargs  # Pass through any additional configuration
            )

            # Save the markdown content
            output_file.write_text(md_text, encoding="utf-8")

            logger.success(f"Successfully converted to markdown: {output_path}")

        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            raise

    def parse_document_to_json(
        self, input_path: str, output_path: str, **kwargs
    ) -> None:
        """
        Parse PDF document to JSON format with enhanced extraction.
        Extracts structured data optimized for downstream LLM processing.
        """
        # Use the parent PyMuPDFProcessor for JSON conversion
        # as PyMuPDF4LLM focuses on markdown optimization
        self.pymupdf_processor.parse_document_to_json(input_path, output_path, **kwargs)

        logger.info(f"Using PyMuPDFProcessor for JSON conversion: {input_path}")

    def chunks_document(
        self,
        input_path: str,
        strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.SEMANTIC,
        chunk_size: int = 1000,
        overlap: int = 100,
        **kwargs,
    ) -> List[DocumentChunk]:
        """
        Chunk a PDF document using LLM-optimized strategies.
        Like providing properly sized portions of text for efficient LLM digestion.

        Default to SEMANTIC strategy for better contextual chunking.
        """
        # Convert string to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = ChunkingStrategy(strategy)
            except ValueError:
                logger.warning(
                    f"Unknown strategy '{strategy}', falling back to SEMANTIC"
                )
                strategy = ChunkingStrategy.SEMANTIC

        logger.info(f"Chunking document with {strategy.value} strategy: {input_path}")

        chunks = []

        try:
            if strategy == ChunkingStrategy.SEMANTIC:
                # Use PyMuPDF4LLM for semantic chunking via markdown
                md_text = pymupdf4llm.to_markdown(input_path)
                chunks = self._chunk_markdown_semantically(md_text, chunk_size, overlap)
            else:
                # Use PyMuPDFProcessor for other chunking strategies
                chunks = self.pymupdf_processor.chunks_document(
                    input_path, strategy, chunk_size, overlap, **kwargs
                )

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
    ) -> Iterator[str]:
        """
        Read PDF documents using PyMuPDF4LLM optimized extraction.
        Unlike the base implementation, this returns markdown strings optimized for LLMs.
        """
        if file_path:
            # Single file
            yield self._read_pdf_to_markdown(file_path)
        elif directory_path:
            # Directory of files
            dir_path = Path(directory_path)
            if not dir_path.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")

            for file in dir_path.rglob("*"):
                if file.suffix.lower() == self.supported_format:
                    yield self._read_pdf_to_markdown(str(file))
        else:
            raise ValueError("Either file_path or directory_path must be provided")

    def _read_pdf_to_markdown(self, file_path: str) -> str:
        """Helper method to read PDF and convert to markdown."""
        try:
            logger.debug(f"Reading PDF to markdown: {file_path}")
            return pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            logger.error(f"Error reading PDF to markdown: {e}")
            raise

    def _chunk_markdown_semantically(
        self, md_text: str, chunk_size: int, overlap: int
    ) -> List[DocumentChunk]:
        """
        Chunk markdown text semantically using LangChain's MarkdownTextSplitter.
        This respects markdown structure (headers, sections) for better semantic chunks.
        """
        # Using LangChain's MarkdownTextSplitter for semantic awareness
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        text_chunks = splitter.split_text(md_text)

        chunks = []
        for idx, text_chunk in enumerate(text_chunks):
            # Try to detect page number from content (heuristic)
            page_number = self._estimate_page_number(text_chunk, md_text)

            chunk = DocumentChunk(
                content=text_chunk,
                page_number=page_number,
                metadata={
                    "strategy": "semantic",
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "markdown_based": True,
                },
                chunk_index=idx,
                chunk_id=f"chunk_{idx:04d}",
            )
            chunks.append(chunk)

        return chunks

    def _estimate_page_number(self, chunk: str, full_text: str) -> int:
        """
        Estimate page number from chunk content by looking for page headers.
        This is a heuristic since PyMuPDF4LLM's markdown doesn't always preserve page numbers.
        """
        # Simple heuristic: count page separators or headers before the chunk
        chunk_position = full_text.find(chunk)
        if chunk_position > 0:
            preceding_text = full_text[:chunk_position]
            # Count lines that likely represent page headers
            import re

            page_markers = re.findall(r"(?:^|\n)#+\s*Page\s*(\d+)", preceding_text)
            if page_markers:
                return int(page_markers[-1])

        return 1  # Default to page 1 if unable to determine

    # @classmethod
    # def to_llama_index_documents(cls, pdf_path: str, **kwargs) -> List[Any]:
    #     """
    #     Convert PDF to LlamaIndex documents directly.
    #     Useful for RAG pipelines that use LlamaIndex.
    #     """
    #     try:
    #         from llama_index.readers.file import PyMuPDFReader
    #     except ImportError:
    #         raise ImportError("llama_index package is required for this feature")

    #     logger.info(f"Converting PDF to LlamaIndex documents: {pdf_path}")
    #     loader = PyMuPDFReader()
    #     documents = loader.load(file_path=pdf_path)

    #     return documents

    @classmethod
    def to_langchain_documents(cls, pdf_path: str, **kwargs) -> List[Any]:
        """
        Convert PDF to LangChain documents directly.
        Useful for RAG pipelines that use LangChain.
        """
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except ImportError:
            raise ImportError(
                "langchain_community package is required for this feature"
            )

        logger.info(f"Converting PDF to LangChain documents: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        return documents


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

    # Initialize processor
    processor = PyMuPDF4LLMProcessor()

    for file in all_files:
        output_path = os.path.join(
            output_path, os.path.basename(file).replace(".pdf", ".md")
        )

        # LLM-optimized markdown conversion with table preservation
        processor.parse_document_to_markdown(file, output_path)

        # # Semantic chunking for better LLM comprehension
        # chunks = processor.chunks_document(
        #     file, strategy="semantic", chunk_size=1000
        # )

        # Read documents as markdown for LLM processing
        # for md_content in processor.read_documents(file_path=file):
        #     # Process markdown content for LLM
        #     pass

        # Convert to RAG frameworks
        langchain_docs = processor.to_langchain_documents(file)
        logger.debug(f"LangChain docs: {langchain_docs}")
        # llama_docs = processor.to_llama_index_documents("sample.pdf")
