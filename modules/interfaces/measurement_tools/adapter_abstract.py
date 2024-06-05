"""
    This file contains the abstract class for the measurement tool adapter.
    The two functions that need to be implemented are initialize and
    measure_statistic. Initialize is used to initialize the measurement tool
    if necessary and measure_statistic is used to measure the statistic of the
    code. This function recieves the filename of the code to measure the
    statistic of the code inside the file.
"""

from abc import ABC, abstractmethod

class MeasurementToolAdapter(ABC):
    """
    Abstract class for the measurement tool.
    """

    @abstractmethod
    def initialize(self)-> int:
        """
        Function to initialize the measurement tool.
        Returns:
            int: 1 if successful, 0 otherwise
        """
        pass

    @abstractmethod
    def measure_statistics(self, command: str)-> str:
        """
        Function to measure the statistic of the code.
        Args:
            command (str): The command to measure the statistic of.
        Returns:
            list with one or multiple dictionarie(s): Keys are 'type', 'mean', 'standard_deviation', 'unit'. The type is the type of the
            statistic, the value is the value of the statistic and the unit is the unit of the statistic. On error, the function should return an empty dictionary. Multiple dictionaries can be returned if multiple statistics are measured. On failure the value of the statistic should be None.
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        Function to get the name of the measurement tool.
        Returns:
            str: The name of the measurement tool.
        """
        pass