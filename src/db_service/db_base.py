from abc import ABC, abstractmethod
from typing import List, Optional


class VectorDBBase(ABC):
    """
    Abstract base class for vector databases.
    Defines the interface for storing, indexing, and querying vector embeddings.
    """

    @abstractmethod
    def connect(self, **kwargs) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def add_documents(self, **kwargs) -> None:
        pass

    @abstractmethod
    def query(self, **kwargs):
        pass

    @abstractmethod
    def get_by_ids(self, **kwargs):
        pass

    @abstractmethod
    def delete(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_collection_info(self, **kwargs):
        pass
