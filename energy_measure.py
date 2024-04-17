import subprocess

"""
    Producer: Tom Cappendijk
    Date: 2024-04-11

    This file contains the EnergyMeasure class. This class is used to measure
    the energy consumed by a program and the time elapsed. The energy is
    measured in Joules and the time is measured in seconds. The tool used to
    measure the energy is perf. This means that perf must be installed on the
    system where this class is used. The perf version used is 6.5.13.

    Note:
    - The event used to measure the energy is power/energy-pkg/. This event
    is linux specific and you need to have the correct permissions to access.
    It is possible that you need to use another event to measure the energy.
    Perf events can be found using the command 'perf list'.

    Disclaimer:
    - The power/energy-pkg/ event is a system wide event. This means that the
    energy consumed by all the processes running on the system is measured.
"""


class EnergyMeasure:
    def __init__(self, filename, interpreter="python3"):
        self.filename = filename
        self.python_interpreter = interpreter

    def measure_energy_extime(self):
        """
        This method measures the energy consumed by a program and the time
        elapsed. The energy is measured in Joules and the time is measured in
        seconds.

        Returns:
            dict: A dictionary containing the output of the program, the energy
            consumed in Joules and the time elapsed in seconds.
        """
        command = f"perf stat -e power/energy-pkg/ \
                       {self.python_interpreter} {self.filename}"
        result = subprocess.run(command, shell=True,
                                capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception("An error occurred while measuring the energy")

        # perf returns the output in stderr
        perf_output = result.stderr

        joules = None
        seconds = None
        for line in perf_output.split('\n'):
            if "Joules" in line:
                joules = float(line.split()[0].replace(',', '.'))
            elif "seconds time elapsed" in line:
                seconds = float(line.split()[0].replace(',', '.'))

        return {"File_output": result.stdout, "Energy consumed": joules,
                "Time elapsed": seconds}
