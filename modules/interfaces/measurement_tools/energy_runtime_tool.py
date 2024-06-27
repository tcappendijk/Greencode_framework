from modules.interfaces.measurement_tools.adapter_abstract import MeasurementToolAdapter
import subprocess
import numpy as np


class EnergyRunTime(MeasurementToolAdapter):

    def __init__(self):

        # Idle energy is the energy consumed by the system when it is idle in Joules per second
        self.__idle_energy = -1

    def initialize(self, time=1) -> int:
        """
            Returns: 1 if successful, 0 otherwise

        """
        command = "perf stat --timeout " + str(int(time * 1000)) + " -e power/energy-pkg/"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            return 0

        idle_energy = -1
        seconds_elapsed = -1
        for line in result.stderr.split('\n'):
            if "Joules" in line:
                idle_energy = float(line.split()[0].replace('.', '').replace(',', '.'))
            if "seconds time elapsed" in line:
                seconds_elapsed = float(line.split()[0].replace('.', '').replace(',', '.'))

        if idle_energy == -1 or seconds_elapsed == -1:
            return 0

        self.__idle_energy = idle_energy / seconds_elapsed
        return 1

    def measure_statistics(self, command: str, repeat = 50) -> str:

        if self.__idle_energy == -1:
            print("Idle energy is not initialized")
            return {}

        energy_command = "perf stat -e power/energy-pkg/ " + command

        energy_values = []
        runtime_values = []

        for _ in range(repeat):
            result = subprocess.run(energy_command, shell=True,
                                    capture_output=True, text=True)

            if result.returncode != 0:
                print("An error occurred while measuring the energy and runtime of command: " + command)
                return [{'type': 'energy', 'values': None, 'unit': 'Joules per second'}, {'type': 'runtime', 'values': None, 'unit': 'seconds'}]

            energy = -1
            runtime = -1
            for line in result.stderr.split('\n'):
                if "Joules" in line:
                    energy = float(line.split()[0].replace('.', '').replace(',', '.'))
                if "seconds time elapsed" in line:
                    runtime = float(line.split()[0].replace('.', '').replace(',', '.'))

            if energy == -1 or runtime == -1:
                print("An error occurred while measuring the energy or runtime of command: " + command)
                return [{'type': 'energy', 'values': None, 'unit': 'Joules per second'}, {'type': 'runtime', 'values': None, 'unit': 'seconds'}]

            energy_values.append(energy)
            runtime_values.append(runtime)

        energy_tool_dict = {'type': 'energy', 'values': energy_values, 'idle_energy': self.__idle_energy, 'unit': 'Joules per second'}
        runtime_tool_dict = {'type': 'runtime', 'values' : runtime_values, 'unit': 'seconds'}
        return [energy_tool_dict, runtime_tool_dict]

    def get_name(self):
        return "Energy"
