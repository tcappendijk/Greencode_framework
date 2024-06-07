from modules.interfaces.measurement_tools.adapter_abstract import MeasurementToolAdapter
import subprocess
import numpy as np


class Energy(MeasurementToolAdapter):

    def __init__(self):
        pass

    def initialize(self) -> int:
        """
            Returns: 1 if successful, 0 otherwise

        """
        return 1

    def measure_statistics(self, command: str, repeat = 50) -> str:

        # Measure the peak memory used by the command

        memory_command = "/usr/bin/time -v " + command
        memory_values = []
        for _ in range(repeat):
            result = subprocess.run(memory_command, shell=True,
                        capture_output=True, text=True)
            if result.returncode != 0:
                print("An error occurred while measuring the memory of command: " + command)
                return [{'type': 'memory', 'values': None, 'unit': 'KB'}]
            for line in result.stderr.split('\n'):
                if "Maximum resident set size" in line:
                    memory_values.append(int(line.split()[5].replace(',', '')))

        tool_dict = {'type': 'memory', 'values': memory_values, 'unit': 'KB'}

        return [tool_dict]

    def get_name(self):
        return "Memory"