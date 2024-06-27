"""
    This program tests the noise in the energy measurements of the measurement
    tool Perf. The program measures the energy consumption of a target function
    with a certain number of iterations. The results are stored in a shelve file
    with the following keys:
    - base_energy: Energy consumption of the base function.
    - base_times: Execution time of the base function.
    - energy: Energy consumption of the target function.
    - times: Execution time of the target function.

    The output directory is specified by the variable output_dir. The default
    value is 'output_noisetest'.
"""

import subprocess
import numpy as np
import shelve
import time
import os

from scipy import stats

number_of_executions_per_iteration_value = 50
number_of_iterations = 1000
base_number_of_operations = 1
best_of = 10

def one_sided_mann_whitney_test(data1, data2, alpha=0.05):
    """
    Perform a one-sided Mann-Whitney U test to determine if data1 is significantly larger than data2. If data1 is not
    significantly larger than data2, then perform the test to determine if data2 is significantly larger than data1.

    Args:
    data1 (array-like): First dataset.
    data2 (array-like): Second dataset.
    alpha (float, optional): Significance level for the test. Default is 0.05.

    Returns:
    int: 1 if data1 is significantly larger than data2,
        -1 if data1 is significantly smaller than data1,
         0 if neither condition is met so it is Unknown wheter data1 is larger
         or smaller but data1 and data2 are from the same distribution,
        "Error" if data1 is smaller and larger,
        "Undifined" if the test is undifined, so data1 is not larger or smaller,
        None if data1 or data2 is None.
    """

    if data1 is None or data2 is None:
        return None

    # Perform Mann-Whitney U test
    _, p_value1 = stats.mannwhitneyu(data1, data2, alternative='greater')
    _, p_value2 = stats.mannwhitneyu(data1, data2, alternative='less')

    # Determine results based on p-values
    if p_value1 < alpha and p_value2 < alpha:
        return "Stochastically equal"
    elif p_value1 < alpha:
        return 1  # data1 is significantly larger than data2
    elif p_value2 < alpha:
        return -1  # data1 is significantly smaller than data1
    else:
        _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p_value < alpha:
            return "Undefined"
        else:
            return 0



def executeTest(iterations):
    result = subprocess.run(['perf', 'stat','-e power/energy-pkg/', 'python3.9', 'targetfunction.py', str(iterations)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result

def retrieveEnergyFromResult(result):
    return float(result.stderr.decode().split()[6])

def retrieveExecutionTimeFromResult(result):
    return float(result.stderr.decode().split()[9])

def runTestBatch(best_of, number_of_executions_per_iteration_value, base_number_of_operations, number_of_iterations, output_dir='output_noisetest'):
    starttime = time.asctime()
    for count in range(best_of):
        # Actually we are working the wrong way around becuase normally the base measurement would be the most energy consuming one.
        # Measure the base values with only one iteration.
        base_energy = np.array([])
        base_times = np.array([])
        for _ in range(number_of_executions_per_iteration_value):
            result = executeTest(base_number_of_operations)
            base_energy = np.append(base_energy, retrieveEnergyFromResult(result))
            base_times = np.append(base_times, retrieveExecutionTimeFromResult(result))

        # Measure a more expensive execution with more iterations.
        energy = np.array([])
        times = np.array([])
        for _ in range(number_of_executions_per_iteration_value):
            result = executeTest(number_of_iterations)
            energy = np.append(energy, retrieveEnergyFromResult(result))
            times = np.append(times, retrieveExecutionTimeFromResult(result))

        # Run Mann Whitney U test.
        print(one_sided_mann_whitney_test(base_energy, energy))
        print(one_sided_mann_whitney_test(base_times, times))

        sname = "shelve_" + starttime + "_" + str(count) + "_" + str(best_of) + "_" + str(base_number_of_operations) + "_" + str(number_of_iterations) + "_" + str(number_of_executions_per_iteration_value) + "_" + str(one_sided_mann_whitney_test(base_energy, energy)) + "_" + str(one_sided_mann_whitney_test(base_times, times))

        # Check if the output directory exists.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sname = os.path.join(output_dir, sname)
        print(sname)
        sf = shelve.open(sname)
        sf['base_energy'] = base_energy
        sf['base_times'] = base_times
        sf['energy'] = energy
        sf['times'] = times
        sf.close()

noepiv = [50, 500]
noi = [1, 10, 100, 1000, 10000, 100000, 1000000]
#offsets = [0, 10, 100, 1000, 10000]

for number_of_executions_per_iteration_value in noepiv:
    for number_of_iterations in noi:
        runTestBatch(best_of, number_of_executions_per_iteration_value, base_number_of_operations, number_of_iterations)
