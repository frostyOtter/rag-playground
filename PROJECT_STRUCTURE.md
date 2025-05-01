# Current Project Structure

```
rag_application/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Application settings and configuration
│   │   └── logging.py           # Loguru logger configuration
│   │
│   ├── data_service/
│   │   ├── __init__.py
│   │   ├── data_base.py              # Abstract base class for preprocessors
│   │   ├── pdf_processor.py     # PyMuPDF implementation
│   │   ├── html_processor.py    # BeautifulSoup implementation
│   │   ├── text_splitter.py     # Text chunking logic
│   │   └── factory.py           # Factory pattern for processor selection
│   │
│   ├── db_service/
│   │   ├── __init__.py
│   │   ├── db_base.py              # Abstract base class for vector databases
│   │   ├── lancedb_impl.py      # LanceDB implementation with search strategies
│   │   ├── chromadb_impl.py     # ChromaDB implementation with search strategies
│   │   ├── pinecone_impl.py     # Pinecone implementation with search strategies
│   │   └── factory.py           # Factory pattern for DB selection
│   │
│   ├── llm_service/
│   │   ├── __init__.py
│   │   ├── llm_base.py              # Abstract base class for LLM providers
│   │   ├── openai_impl.py       # OpenAI implementation
│   │   ├── anthropic_impl.py    # Anthropic implementation
│   │   ├── gemini_impl.py       # Google Gemini implementation
│   │   └── factory.py           # Factory pattern for LLM selection
│   │
│   ├── embedding_service/
│   │   ├── __init__.py
│   │   ├── embedding_base.py              # Abstract base class for embeddings
│   │   ├── openai_impl.py       # OpenAI embeddings
│   │   ├── cohere_impl.py       # Cohere embeddings
│   │   ├── local_impl.py        # Local embeddings (e.g., sentence-transformers)
│   │   └── factory.py           # Factory pattern for embedding selection
│   │
│   ├── reranker_service/
│   │   ├── __init__.py
│   │   ├── reranker_base.py              # Abstract base class for rerankers
│   │   ├── cohere_impl.py       # Cohere reranker
│   │   ├── cross_encoder_impl.py # CrossEncoder reranker
│   │   ├── colbert_impl.py      # Colbert reranker
│   │   └── factory.py           # Factory pattern for reranker selection
│   │
│   ├── rag_service/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main RAG pipeline orchestrator
│   │   ├── retriever.py         # Document retrieval logic
│   │   └── generator.py         # Response generation logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py          # Document data model
│   │   ├── query.py             # Query data model
│   │   └── response.py          # Response data model
│   │
│   └── utils/
│       ├── __init__.py
│       ├── async_helpers.py     # Async utilities
│       ├── data_validation.py   # Input validation utilities
│       └── exceptions.py        # Custom exceptions
│
├── .env.example                # Environment variables template
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```