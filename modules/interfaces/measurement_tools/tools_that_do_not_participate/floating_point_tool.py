from modules.interfaces.measurement_tools.adapter_abstract import MeasurementToolAdapter
import subprocess


class Energy(MeasurementToolAdapter):

    def __init__(self):
        pass

    def initialize(self) -> int:
        """
            Returns: 1 if successful, 0 otherwise

        """
        return 1

    def measure_statistics(self, command: str) -> str:

        # Measure the number of floating point operations

        flop_command = "perf stat -e fp_ret_sse_avx_ops.all " + command
        result = subprocess.run(flop_command, shell=True,
                                capture_output=True, text=True)

        if result.returncode != 0:
            print("An error occurred while measuring the floating point operations of command: " + command)
            return [{'type': 'flops', 'values': None, 'unit': 'flops'}]

        flops = -1
        for line in result.stderr.split('\n'):
            if "fp_ret_sse_avx_ops.all" in line:
                flops = int(line.split()[0].replace(',', ''))

        if flops == -1:
            print("An error occurred while measuring the floating point operations of command: " + command)
            return [{'type': 'flops', 'values': None, 'unit': 'flops'}]

        tool_dict = {'type': 'flops', 'values':[flops], 'unit': 'flops'}
        return [tool_dict]

    def get_name(self):
        return "Floating Point Operations"
