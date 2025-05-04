from abc import ABC, abstractmethod


class DataProcessorBase(ABC):
    """
    Abstract base class for data processors.
    Defines the interface for processing various data formats into a standard Document format.
    """

    @abstractmethod
    def parse_document_to_markdown(self, input_path: str, output_path: str, **kwargs):
        """
        Parse unstructured data into a structured Document format.
        After that save it to markdown file format.
        """
        pass

    @abstractmethod
    def parse_document_to_json(self, input_path: str, output_path: str, **kwargs):
        """
        Parse unstructured data into a structured Document format.
        After that save it to json file format.
        """
        pass

    @abstractmethod
    def chunks_document(self, **kwargs):
        """
        Apply chunks methods to a Document.
        """
        pass

    @abstractmethod
    def read_documents(self, **kwargs):
        """
        Read unstructured documents from a specified file format.
        """
        pass
