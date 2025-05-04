from abc import ABC, abstractmethod
from typing import List


class LLMProviderBase(ABC):
    """
    Abstract base class for Language Model providers.
    Defines the interface for interacting with different LLM services.
    """

    @abstractmethod
    def generate_response(self, **kwargs):
        pass

    @abstractmethod
    def get_model_info(self, **kwargs) -> dict:
        pass
