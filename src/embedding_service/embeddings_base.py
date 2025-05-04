from abc import ABC, abstractmethod
from typing import List


class EmbeddingBase(ABC):
    """
    Abstract base class for embedding providers.
    Defines the interface for generating vector embeddings for text.
    """

    @abstractmethod
    def generate_embeddings(self, text_list: List[str], **kwargs) -> List[List[float]]:
        pass

    @abstractmethod
    def generate_embedding_from_text(self, text: str, **kwargs) -> List[float]:
        pass

    @abstractmethod
    def get_embedding_dimensions(self, **kwargs) -> int:
        pass
