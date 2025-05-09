# RAG Application

A modular Retrieval-Augmented Generation (RAG) application that provides flexible document processing, LLM integration, and vector database capabilities.

## Features

- Document processing with PyMuPDF
- LLM integration with Gemini
- Vector database support with Neo4j
- Support for PDF document ingestion
- Flexible single file or batch processing

## Prerequisites

- Python 3.x
- Neo4j Database
- Required Python packages (see requirements.txt)

## Usage

### 1. Document Processing

Process documents using either of the following methods:

```bash
# Using PyMuPDF SDK
./Scripts/run_pymupdf_processor.sh

# Using PyMuPDF for LLM processing
./Scripts/run_pymupdf4llm.sh
```

**Note:** Ensure to configure input and output paths before running the scripts.

### 2. LLM Service

Run the Gemini LLM service:

```bash
./Scripts/run_gemini_llm_service.sh
```

This service supports structured output generation.

### 3. Neo4j Database Operations

The Neo4j service supports two main operations:

```bash
Scripts/run_neo4j_service.shingest
```
Check to uncomment service
argv 1 = ingest -> argv 2 = path to file or folder
supported file type: .pdf

argv 1 = query -> argv 2 = message to do rag

**Supported File Types:**
- PDF (.pdf)

## Configuration

Before running the services, ensure to:
0. Setup UV
1. Configure your Neo4j database connection
2. Set up appropriate input/output paths
3. Configure LLM API keys if required

