import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from typing import List, Optional, Literal
import cohere
from loguru import logger

from src.embedding_service.embeddings_base import EmbeddingBase


class CohereEmbeddingService(EmbeddingBase):

    # Model configurations - mapping model names to their dimensions
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: Literal[
            "search_document", "search_query", "classification", "clustering", "image"
        ] = "search_document",
        embedding_types: Optional[List[str]] = None,
    ):
        """
        Initialize the Cohere Embedding Service.

        Args:
            model: The Cohere model to use for embeddings
            api_key: Cohere API key. If None, uses COHERE_API_KEY from environment
            input_type: Type of input being embedded
            embedding_types: Types of embeddings to return. Defaults to ["float"]
        """
        # Get API key from environment if not provided
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Cohere API key must be provided in COHERE_API_KEY environment variable"
            )

        # Validate and set model
        if model not in self.MODEL_DIMENSIONS:
            logger.warning(
                f"Unknown model {model}. Using default embed-english-v3.0. "
                f"Known models: {list(self.MODEL_DIMENSIONS.keys())}"
            )
            self.model = "embed-english-v3.0"
        else:
            self.model = model

        self.input_type = input_type
        self.embedding_types = embedding_types or ["float"]

        # Initialize Cohere client
        try:
            self.client = cohere.ClientV2(api_key=self.api_key)
            logger.info(f"Initialized Cohere client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise

    def generate_embeddings(self, text_list: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere's API.

        Optimized for batch processing - Cohere API supports up to 96 texts per request.
        For lists exceeding this limit, texts are processed in optimal batches.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional arguments passed to the Cohere API

        Returns:
            List of embedding vectors as lists of floats
        """
        if not text_list:
            logger.warning("Empty text list provided to generate_embeddings")
            return []

        # Override defaults with kwargs if provided
        model = kwargs.get("model", self.model)
        input_type = kwargs.get("input_type", self.input_type)
        embedding_types = kwargs.get("embedding_types", self.embedding_types)

        # Cohere API can handle up to 96 texts per request
        BATCH_SIZE = 96
        all_embeddings = []
        from tqdm import tqdm

        try:
            # Process in optimal batch sizes
            for i in tqdm(
                range(0, len(text_list), BATCH_SIZE), desc="Generating embeddings"
            ):
                batch = text_list[i : i + BATCH_SIZE]

                response = self.client.embed(
                    texts=batch,
                    model=model,
                    input_type=input_type,
                    embedding_types=embedding_types,
                )

                # Extract float embeddings (primary embedding type)
                if hasattr(response.embeddings, "float_") or hasattr(
                    response.embeddings, "float"
                ):
                    if hasattr(response.embeddings, "float_"):
                        all_embeddings.extend(response.embeddings.float_)
                    else:
                        all_embeddings.extend(response.embeddings.float)
                else:
                    raise ValueError(
                        "Cannot find 'float_' or 'float' in response.embeddings, recheck the SDK version."
                    )

            logger.debug(f"Successfully embedded {len(text_list)} texts")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_embedding_from_text(self, text: str, **kwargs) -> List[float]:

        embeddings = self.generate_embeddings([text], **kwargs)
        return embeddings[0] if embeddings else []

    def get_embedding_dimensions(self, **kwargs) -> int:
        """
        Get the dimension size of the embeddings for the current model.

        Args:
            **kwargs: Not used, but included for interface compatibility

        Returns:
            Dimension size of the embeddings
        """
        model = kwargs.get("model", self.model)
        dimensions = self.MODEL_DIMENSIONS.get(model)

        if dimensions is None:
            logger.warning(
                f"Unknown dimensions for model {model}. "
                f"You may need to update MODEL_DIMENSIONS dictionary."
            )
            # Reasonable default for most models
            return 1024

        return dimensions


# Example usage demonstrating the flexibility and simplicity of the design
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables")
    # Basic usage with defaults
    service = CohereEmbeddingService(api_key=cohere_api_key)

    # Generate embeddings for multiple texts
    texts = ["Hello world", "Machine learning is exciting"]
    embeddings = service.generate_embeddings(texts)
    logger.info(f"Embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
    # Generate single embedding
    single_embedding = service.generate_embedding_from_text("Single text example")
    logger.info(f"Single embedding length: {len(single_embedding)}")
    # Get embedding dimensions
    dimensions = service.get_embedding_dimensions()
    logger.info(f"Model dimensions: {dimensions}")

    # # Example with custom configurations
    # custom_service = CohereEmbeddingService(
    #     model="embed-multilingual-v3.0",
    #     input_type="search_query",
    #     embedding_types=["float", "int8"],
    # )

    # # Search-query focused embeddings
    # query_embedding = custom_service.generate_embedding_from_text(
    #     "Find Python developers"
    # )
    # logger.info(f"Query embedding length: {len(query_embedding)}")
