"""
    This file is used to compare the base and optimized prompt statistics
    for the code problems Sort_List, Assign_Cookies, and
    Median_of_Two_Sorted_Arrays. The metrics used for comparison are
    similarity, energy, memory, flops, and runtime. The similarity is
    calculated using pycode-similar. The energy, memory, flops, and runtime
    are retrieved from the statistics of the base and optimized prompts. The
    plot shows the percentage difference between the base and optimized
    prompts for the metrics mentioned above.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import subprocess
import re
import json


def plot(data, x_label_values, y_label_values, x_label, y_label, plot_title, filename=None, right_shift=0.0, up_shift=0.0, cbar=False, cmap='viridis', vmin=None, vmax=None, cbar_kws=None):
    fig, ax = plt.subplots(figsize=(24, 12))  # Adjust the figsize as needed

    # Use a white colormap and annotate the values
    heatmap = sns.heatmap(data, cmap=sns.color_palette(["white"]), fmt=".1f", cbar=cbar, ax=ax, square=True,
                          vmin=vmin, vmax=vmax, annot=True, annot_kws={"color": "black"}, linewidths=1, linecolor='black')

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
        plt.savefig(f'plots/'+ plot_title.replace(" ", "_") + '_' + x_label + '_' + y_label + '.png', bbox_inches='tight')
    else:
        plt.savefig(f'plots/'+ filename, bbox_inches='tight')



def process_output_list_with_dictionaries(output_list_with_dictionaires):

    output_dictionary = {}

    code_problems = ['Sort_List', 'Assign_Cookies', 'Median_of_Two_Sorted_Arrays']
    optimization_labels = ['energy', 'library_functions', 'for_loop']


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
            if optimization_label in prompt_label:
                optimization_label_found = True
                output_dictionary[code_problem][model_name][optimization_label] = {}
                output_dictionary[code_problem][model_name][optimization_label]['statistics'] = output_dict['statistics']
                output_dictionary[code_problem][model_name][optimization_label]['filename'] = output_dict['filename']

        if not optimization_label_found:
            output_dictionary[code_problem][model_name]['base'] = {}
            output_dictionary[code_problem][model_name]['base']['statistics'] = output_dict['statistics']
            output_dictionary[code_problem][model_name]['base']['filename'] = output_dict['filename']

    return output_dictionary


def calculate_similarity_between_files(reference_file_name, candidate_file_name):
    """Calculate similarity between two files using pycode-similar and return the percentage."""

    result = subprocess.run(
        ['pycode_similar', reference_file_name, candidate_file_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("Error running pycode_similar:", result.stderr)
        return None

    output = result.stdout

    match = re.search(r'(\d+\.\d+) %', output)
    if match:
        percentage = float(match.group(1))
        return percentage
    else:
        print("Similarity percentage not found in the output.")
        return None

def calculate_percentage_difference(base_values, candidate_values):

    if base_values == None or candidate_values == None:
        return None

    base_value_mean = np.mean(base_values)
    candidate_value_mean = np.mean(candidate_values)

    return (candidate_value_mean - base_value_mean) / base_value_mean * 100


def process_data_for_plot(output_dictionary):

    for code_problem in output_dictionary.keys():
        data = np.zeros((15, 5))
        y_label_values = []
        x_label_values = ['similarity', 'energy', 'memory', 'flops', 'runtime']
        for model_name in output_dictionary[code_problem].keys():
            base_prompt_statistics = output_dictionary[code_problem][model_name]['base']['statistics']
            base_prompt_filename = output_dictionary[code_problem][model_name]['base']['filename']

            for optimization_label in output_dictionary[code_problem][model_name].keys():
                if optimization_label == 'base':
                    continue

                y_label_values.append(model_name + ' ' + optimization_label.replace('_', ' '))


                similarity = calculate_similarity_between_files(base_prompt_filename, output_dictionary[code_problem][model_name][optimization_label]['filename'])
                energy = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'energy'][0]['values'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'energy'][0]['values'])

                memory = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'memory'][0]['values'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'memory'][0]['values'])

                flopst = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'flops'][0]['values'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'flops'][0]['values'])

                runtime = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'runtime'][0]['values'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'runtime'][0]['values'])

                data[y_label_values.index(model_name + ' ' + optimization_label.replace('_', ' ')), x_label_values.index('similarity')] = similarity
                data[y_label_values.index(model_name + ' ' + optimization_label.replace('_', ' ')), x_label_values.index('energy')] = energy
                data[y_label_values.index(model_name + ' ' + optimization_label.replace('_', ' ')), x_label_values.index('memory')] = memory
                data[y_label_values.index(model_name + ' ' + optimization_label.replace('_', ' ')), x_label_values.index('flops')] = flopst
                data[y_label_values.index(model_name + ' ' + optimization_label.replace('_', ' ')), x_label_values.index('runtime')] = runtime

        plot(data, x_label_values, y_label_values, 'Metrics', 'Models', code_problem.replace('_', ' '), cbar=False)

if __name__ == '__main__':
    dir = 'output_run0'
    output_filename='output_with_statistic.json'

    with open(dir + '/' + output_filename, 'r') as file:
        output_list_with_dictionaires = json.load(file)

    output_dictionary = process_output_list_with_dictionaries(output_list_with_dictionaires)

    process_data_for_plot(output_dictionary)
