# Current project diagram
```
graph TB
    subgraph "RAG Application Architecture"
        subgraph "Core Services"
            RAG[RAG Service] --> DataService
            RAG --> DBService
            RAG --> LLMService
            RAG --> EmbeddingService
            RAG --> RerankerService
        end
        
        subgraph "Data Service"
            DataService[Data Service Interface]
            DataService --> PreprocessorBase["Preprocessor Base Class"]
            PreprocessorBase --> PDFProcessor["PDF Processor"]
            PreprocessorBase --> PyPDFProcessor["PyPDF Processor"]
            PreprocessorBase --> HTMLProcessor["HTML Processor"]
        end
        
        subgraph "DB Service"
            DBService[DB Service Interface]
            DBService --> VectorDBBase["Vector DB Base Class"]
            VectorDBBase --> LanceDB["LanceDB"]
            VectorDBBase --> ChromaDB["ChromaDB"]
            VectorDBBase --> Pinecone["Pinecone"]
        end
        
        subgraph "LLM Service" 
            LLMService[LLM Service Interface]
            LLMService --> LLMProviderBase["LLM Provider Base Class"]
            LLMProviderBase --> OpenAI["OpenAI"]
            LLMProviderBase --> Anthropic["Anthropic"]
            LLMProviderBase --> Gemini["Gemini"]
        end
        
        subgraph "Embedding Service"
            EmbeddingService[Embedding Service Interface]
            EmbeddingService --> EmbeddingBase["Embedding Provider Base Class"]
            EmbeddingBase --> OpenAIEmbedding["OpenAI Embeddings"]
            EmbeddingBase --> CohereEmbedding["Cohere Embeddings"]
            EmbeddingBase --> LocalEmbedding["Local Embeddings"]
        end
        
        subgraph "Reranker Service"
            RerankerService[Reranker Service Interface]
            RerankerService --> RerankerBase["Reranker Base Class"]
            RerankerBase --> CohereReranker["Cohere Reranker"]
            RerankerBase --> CrossEncoder["CrossEncoder"]
            RerankerBase --> Colbert["Colbert"]
        end
        
        subgraph "Client Layer"
            Client["Client Application"] --> RAG
        end
    end
    
    style RAG fill:#ff9999
    style Client fill:#99ccff
    style PreprocessorBase fill:#ffffcc
    style VectorDBBase fill:#ffffcc
    style LLMProviderBase fill:#ffffcc
    style EmbeddingBase fill:#ffffcc
    style RerankerBase fill:#ffffcc
```