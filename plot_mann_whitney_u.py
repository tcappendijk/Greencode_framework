"""
    This program calculates the Mann-Whitney U test for the energy measurements
    of the different models. The program reads the energy measurements from run
    0 and calculates the Mann-Whitney U test for the energy measurements of the
    different models. The results are plotted in a heatmap. The results are
    plotted per code problem and for all code problems combined.
"""

import numpy as np
from scipy import stats
import json
from matplotlib import pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

def process_output_list_with_dictionaries(output_list_with_dictionaires):
    """
    Process the output list with dictionaries to a dictionary with the following structure:
    {
        code_problem: {
            model_name: {
                optimization_label: {
                    statistics: List of dictionaries with the statistics of the model,
                    filename: Filename of the model
                }
            }
        }
    }

    Args:
        output_list_with_dictionaires (list): List with dictionaries containing the
        output of the measurement tools.
    Returns:
        dict: Dictionary with the structure described above.
    """

    output_dictionary = {}

    code_problems = ['Sort_List', 'Assign_Cookies', 'Median_of_Two_Sorted_Arrays']
    optimization_labels = ['energy', 'library functions', 'for loop']


    for output_dict in output_list_with_dictionaires:
        model_name = output_dict['model_name']

        code_problem = ''
        prompt_label = output_dict['prompt_label']

        for code_problem_name in code_problems:
            if code_problem_name in prompt_label:
                code_problem = code_problem_name
                break

        if code_problem not in output_dictionary:
            output_dictionary[code_problem] = {}


        if model_name not in output_dictionary[code_problem]:
            output_dictionary[code_problem][model_name] = {}

        optimization_label_found = False
        for optimization_label in optimization_labels:
            if optimization_label.replace(' ', '_') in prompt_label:
                optimization_label_found = True
                output_dictionary[code_problem][model_name][optimization_label] = {}
                output_dictionary[code_problem][model_name][optimization_label]['statistics'] = output_dict['statistics']
                output_dictionary[code_problem][model_name][optimization_label]['filename'] = output_dict['filename']

        if not optimization_label_found:
            output_dictionary[code_problem][model_name]['base'] = {}
            output_dictionary[code_problem][model_name]['base']['statistics'] = output_dict['statistics']
            output_dictionary[code_problem][model_name]['base']['filename'] = output_dict['filename']

    return output_dictionary


def one_sided_mann_whitney_test(data1, data2, alpha=0.01):
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
         "Stochastically Equal" if data1 and data2 are not from the same distribution
         but one is also not bigger than the other,
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
        return "Stochastically Equal"
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

def plot(data, annotations, x_label_values, y_label_values, x_label, y_label, plot_title, filename=None, right_shift=0.0, up_shift=0.0, cbar=False, cmap='viridis', vmin=None, vmax=None, cbar_kws=None):
    """
    Plot a heatmap with the given data and annotations. The plot is saved in the
    plots folder.

    Args:
        data: Data for the plot.
        annotations: Annotations for the plot.
        x_label_values: Values for the x-axis labels.
        y_label_values: Values for the y-axis labels.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        plot_title: Title for the plot.
        filename: Filename for the plot. If None, the filename is generated
        based on the plot_title, x_label and y_label.
    """

    fig, ax = plt.subplots(figsize=(24, 12))

    heatmap = sns.heatmap(data, annot=annotations, cmap=cmap, fmt="", cbar=cbar, ax=ax, square=True, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws)

    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_xticklabels(x_label_values, ha='center', rotation=90)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_yticklabels(y_label_values, va='center', rotation=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if plot_title != "":
        ax.set_title(plot_title.replace("_", " "))

    pos = ax.get_position()
    ax.set_position([pos.x0 + right_shift, pos.y0 + up_shift, pos.width, pos.height])

    if cbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_xlabel('Values', labelpad=20)
        cbar.ax.xaxis.set_label_position('top')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    if filename is None:
        plt.savefig(f'plots/'+ plot_title.replace(" ", "_") + '_' + x_label + '_' + y_label, bbox_inches='tight')
    else:
        plt.savefig(f'plots/'+ filename, bbox_inches='tight')

def plot_Mann_Whitney_U_test_problem_all_models_per_problem(output_dictionary):
    """
    Plot the Mann Whitney U test for the energy measurements of the different models
    per code problem. The results are plotted in a heatmap.

    Args:
        output_dictionary: Dictionary with the output of the measurement tools.
    """

    for code_problem in output_dictionary.keys():
        dict_models_output = {}
        for model_name in output_dictionary[code_problem].keys():
            for optimization_label in output_dictionary[code_problem][model_name].keys():
                dict_models_output[model_name + ' ' + optimization_label] = output_dictionary[code_problem][model_name][optimization_label]['statistics']

        # Predefined models list
        models_list = ['deepseek-coder-33b-base base', 'deepseek-coder-33b-base energy', 'deepseek-coder-33b-base library functions', 'deepseek-coder-33b-base for loop', 'deepseek-coder-33b-instruct base', 'deepseek-coder-33b-instruct energy', 'deepseek-coder-33b-instruct library functions', 'deepseek-coder-33b-instruct for loop', 'CodeLlama-70b base', 'CodeLlama-70b energy', 'CodeLlama-70b library functions', 'CodeLlama-70b for loop', 'CodeLlama-70b-Instruct base', 'CodeLlama-70b-Instruct energy', 'CodeLlama-70b-Instruct library functions', 'CodeLlama-70b-Instruct for loop', 'CodeLlama-70b-Python base', 'CodeLlama-70b-Python energy', 'CodeLlama-70b-Python library functions', 'CodeLlama-70b-Python for loop']

        x_labels = models_list
        y_labels = models_list

        # Initialize data and annotations arrays
        data = np.zeros((len(models_list), len(models_list)))
        annotations = np.empty((len(models_list), len(models_list)), dtype=object)

        for i, model1 in enumerate(models_list):
            for j, model2 in enumerate(models_list):
                if i == j:
                    annotations[i][j] = '\\'
                    continue

                energy_model1 = [i['values'] for i in dict_models_output[model1] if i['type'] == 'energy'][0]
                energy_model2 = [i['values'] for i in dict_models_output[model2] if i['type'] == 'energy'][0]

                result = one_sided_mann_whitney_test(energy_model1, energy_model2)

                if result == "Stochastically Equal" or result == "Undifined":
                    annotations[i][j] = result
                else:
                    if result is None:
                        annotations[i][j] = 'N/A'
                    else:
                        data[i][j] = result
                        # Set annotation based on result
                        if result == 1:
                            annotations[i][j] = '+'
                        elif result == -1:
                            annotations[i][j] = '-'
                        elif result == 0:
                            annotations[i][j] = 'Unk'

        plot(data, annotations, x_labels, y_labels, 'Models', 'Models', f'Mann Whitney U test for {code_problem}')


def plot_Mann_Whitney_U_test_all_models_all_problems(output_dictionary):
    """
    Plot the Mann Whitney U test for the energy measurements of the different models
    for all code problems combined. The results are plotted in a heatmap.

    Args:
        output_dictionary: Dictionary with the output of the measurement tools.
    """
    # Predefined models list
    models_list = ['deepseek-coder-33b-base base', 'deepseek-coder-33b-base energy', 'deepseek-coder-33b-base library functions', 'deepseek-coder-33b-base for loop', 'deepseek-coder-33b-instruct base', 'deepseek-coder-33b-instruct energy', 'deepseek-coder-33b-instruct library functions', 'deepseek-coder-33b-instruct for loop', 'CodeLlama-70b base', 'CodeLlama-70b energy', 'CodeLlama-70b library functions', 'CodeLlama-70b for loop', 'CodeLlama-70b-Instruct base', 'CodeLlama-70b-Instruct energy', 'CodeLlama-70b-Instruct library functions', 'CodeLlama-70b-Instruct for loop', 'CodeLlama-70b-Python base', 'CodeLlama-70b-Python energy', 'CodeLlama-70b-Python library functions', 'CodeLlama-70b-Python for loop']

    x_labels = models_list
    y_labels = models_list

    unknown_data_matrix = np.zeros((len(models_list), len(models_list)))
    unknown_annotations = np.empty((len(models_list), len(models_list)), dtype=object)

    known_data_matrix = np.zeros((len(models_list), len(models_list)))
    known_annotations = np.empty((len(models_list), len(models_list)), dtype=object)

    for code_problem in output_dictionary.keys():
        dict_models_output = {}
        for model_name in output_dictionary[code_problem].keys():
            for optimization_label in output_dictionary[code_problem][model_name].keys():
                dict_models_output[model_name + ' ' + optimization_label] = output_dictionary[code_problem][model_name][optimization_label]['statistics']

        for i, model1 in enumerate(models_list):
            for j, model2 in enumerate(models_list):
                if i == j:
                    unknown_annotations[i][j] = '\\'
                    known_annotations[i][j] = '\\'
                    continue
                else:
                    known_annotations[i][j] = ''
                    unknown_annotations[i][j] = ''

                energy_model1 = [i['values'] for i in dict_models_output[model1] if i['type'] == 'energy'][0]
                energy_model2 = [i['values'] for i in dict_models_output[model2] if i['type'] == 'energy'][0]

                result = one_sided_mann_whitney_test(energy_model1, energy_model2)

                if result == "Stochastically Equal" or result == "Undefined":
                    unknown_annotations[i][j] = result
                    unknown_data_matrix[i][j] += 1
                else:
                    if result is None:
                        unknown_annotations[i][j] = 'N/A'
                        unknown_data_matrix[i][j] += 1
                    else:
                        if result == 1 or result == -1:
                            known_data_matrix[i][j] += result
                        elif result == 0:
                            unknown_data_matrix[i][j] += 1

    cmap = LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)
    plot(known_data_matrix, known_annotations, x_labels, y_labels, 'Models', 'Models', "", filename="known_values_all_code_problems", cbar=True, cmap=cmap)
    cmap = LinearSegmentedColormap.from_list('rg',["w", "r"], N=256)
    plot(unknown_data_matrix, unknown_annotations, x_labels, y_labels, 'Models', 'Models', "", filename="unknown_values_all_code_problems", cbar=True, cmap=cmap)



if __name__ == '__main__':
    dir = 'output_run0'
    output_filename='output_with_statistic.json'

    with open(dir + '/' + output_filename, 'r') as file:
        output_list_with_dictionaires = json.load(file)

    output_dictionary = process_output_list_with_dictionaries(output_list_with_dictionaires)
    plot_Mann_Whitney_U_test_problem_all_models_per_problem(output_dictionary)
    plot_Mann_Whitney_U_test_all_models_all_problems(output_dictionary)
