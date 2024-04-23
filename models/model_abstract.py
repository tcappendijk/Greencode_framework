"""
    This file contains the abstract class for the models. The two functions
    that need to be implemented are initialize and handle_code_prompt. Initialize
    is used to initialize the model and handle_code_prompt is used to handle the
    code prompt and return the generated code togheter with the name of the model.
"""


from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract class for the models.
    """

    @abstractmethod
    def initialize(self)-> int:
        """
        Function to initialize the model.
        Returns:
            int: 0 if successful, -1 otherwise
        """
        pass

    @abstractmethod
    def handle_code_prompt(self, code_prompt: str)-> str:
        """
        Function to handle the code prompt.
        Args:
            code_prompt (str): The code prompt to handle.

        Returns:
            str: The generated code.
        """
        pass

    @abstractmethod
    def get_name(self)-> str:
        """
        Function to get the name of the model
        """
        pass

    @abstractmethod
    def close(self)-> int:
        """
        Function to close the model.
        Returns:
            int: 0 if successful, -1 otherwise
        """
        pass