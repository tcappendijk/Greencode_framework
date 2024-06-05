from modules.interfaces.measurement_tools_interface import MeasurementInterface

class MeasureStatistics:

    def __init__(self) -> None:
        self.__measurement_tools_interface = MeasurementInterface()


    def initialize_all_tools(self):
        """
        This function initializes all the measurement tools. On succes, it returns 1,
        otherwise 0.
        """
        if self.__measurement_tools_interface.initialize_all_tools() == 0:
            return 0
        return 1


    def measure_statistics(self, dictionaries):
        """
        This function makes it possible to measure the statistics of the code using all the measurement tools.

        Args:
            dictionaries (list): A list of dictionaries containing the prompt,
            code and filename of the code.
        Returns:
            list: A list of dictionaries containing the type, value and
            unit of the statistic. If the measurement failed, it will not be
            included in the list.
        """

        measured_statistic = []
        for output in dictionaries:
            output['statistics'] = self.__measurement_tools_interface.measure_statistics('python3 ' + output['filename'])
            measured_statistic.append(output)

        return measured_statistic
