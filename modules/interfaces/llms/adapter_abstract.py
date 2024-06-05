"""
    This file contains the abstract class for the adapater of the LLMs. The two functions
    that need to be implemented are initialize and handle_code_prompt. Initialize
    is used to initialize the model and handle_code_prompt is used to handle the
    code prompt and return the generated code togheter with the name of the model.
"""

from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    """
    Abstract class for the models.
    """

    @abstractmethod
    def initialize(self)-> int:
        """
        Function to initialize the model.
        Returns:
            int: 1 if successful, 0 otherwise
        """
        pass

    @abstractmethod
    def process_code_prompt(self, code_prompt: str)-> str:
        """
        Function to process the code prompt by the LLM.
        Args:
            code_prompt (str): The code prompt to handle.

        Returns:
            str: The generated code. Or an empty string if the model could not handle the code prompt.
        """
        pass

    @abstractmethod
    def get_name(self)-> str:
        """
        Function to get the name of the model. The name should match the name of the
        file that the model is implemeted in. This is due to error handling where the model
        is referred to by the name of the file.
        """
        pass

    @abstractmethod
    def close(self)-> int:
        """
        Function to close the model.
        Returns:
            int: 1 if successful, 0 otherwise
        """
        pass