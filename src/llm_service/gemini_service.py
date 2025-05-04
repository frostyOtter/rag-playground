import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from typing import List, Dict, Any, Optional

from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.exceptions import (
    OutputParserException,
)  # Or a more general LangChain/API exception

from src.llm_service.llm_base import LLMProviderBase


class GeminiServiceProvider(LLMProviderBase):
    """
    Implementation of LLMProviderBase using Google's Gemini models via LangChain.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",  # Using a recommended recent model
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        **kwargs: Any,  # To accommodate future base class args or specific Gemini args
    ):

        logger.info(f"Initializing GeminiServiceProvider with model: {model_name}")

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.additional_kwargs = kwargs

        try:
            self.client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                api_key=self.api_key,
                **self.additional_kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI client: {e}")
            raise RuntimeError(f"error: {e}") from e

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Converts a list of {'role': str, 'content': str} dicts to LangChain BaseMessages."""
        langchain_messages = []
        for message_dict in messages:
            role = message_dict.get("role")
            content = message_dict.get("content")

            if not role or not content:
                logger.warning(
                    f"Skipping message due to missing 'role' or 'content': {message_dict}"
                )
                continue  # Skip malformed messages

            role_lower = role.lower()
            if role_lower == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role_lower == "human" or role_lower == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role_lower == "ai" or role_lower == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(
                    f"Unsupported message role '{role}'. Treating as 'human'."
                )
                # Defaulting to HumanMessage might be okay, or raise an error depending on strictness needed
                langchain_messages.append(HumanMessage(content=content))
        return langchain_messages

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generates a response from the Gemini model based on the provided messages.

        Args:
            messages: A list of dictionaries, where each dictionary should have
                      a 'role' (e.g., 'system', 'human', 'ai') and 'content' key.
            **kwargs: Additional keyword arguments to potentially override ChatGoogleGenerativeAI settings
                      for this specific call. Currently unused but kept for interface compliance.


        Returns:
            The generated response content as a string.

        Raises:
            RuntimeError: If the API call fails or input is invalid.
            ValueError: If the messages list is empty or contains invalid entries.
        """
        if not messages:
            logger.error("generate_response called with empty messages list.")
            # Returning "" might hide issues, raising an error is often better.
            raise ValueError("Cannot generate response from empty messages list.")

        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)
            if not langchain_messages:
                logger.error(
                    "Message conversion resulted in an empty list. Check input format/content."
                )
                raise ValueError("No valid messages found after conversion.")

            # Invoke the model
            response = self.client.invoke(
                langchain_messages
            )  # Pass kwargs if needed: self.client.invoke(langchain_messages, **kwargs)
            logger.info(f"Token Usage: {response.usage_metadata}")
            response_content = response.content

            return response_content

        except OutputParserException as ope:
            logger.error(f"Failed to parse the LLM output: {ope}")
            raise RuntimeError(f"LLM output parsing failed: {ope}") from ope
        except Exception as e:
            # Catching a general exception for API/network/LangChain internal errors.
            logger.error(f"Error generating response from Gemini API: {e}")
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    def get_model_info(self, **kwargs) -> dict:
        info = {
            "provider": "google_gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "additional_kwargs": self.additional_kwargs,
        }
        logger.debug(f"Model info: {info}")
        return info


# --- Example Usage (Updated) ---
if __name__ == "__main__":

    user_message = (
        sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    )

    from dotenv import load_dotenv

    load_dotenv()

    API_KEY = os.getenv("GOOGLE_API_KEY")
    # Ensure GOOGLE_API_KEY is set in your environment variables before running

    gemini_provider = GeminiServiceProvider(api_key=API_KEY)

    messages_to_send = [
        {
            "role": "system",
            "content": "You are a helpful assistant that explains technical concepts simply.",
        },
        {
            "role": "human",
            "content": user_message,
        },
    ]

    response_text = gemini_provider.generate_response(messages_to_send)
    logger.info(f"Generated Response:\n{response_text}")
