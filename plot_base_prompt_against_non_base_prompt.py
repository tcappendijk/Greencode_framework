import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import subprocess
import re
import json


def plot(data, x_label_values, y_label_values, x_label, y_label, plot_title, code_problem_name, right_shift=0.5):
    plt.figure(figsize=(16, 9))
    sns.heatmap(data, cmap='viridis', annot=True, fmt=".2f")

    plt.xticks(ticks=np.arange(data.shape[1]) + 0.5, labels=x_label_values, ha='center')
    plt.yticks(ticks=np.arange(data.shape[0]) + 0.5, labels=y_label_values, rotation=0, va='center')

    plt.tick_params(axis='x', which='both', bottom=True, top=False)
    plt.tick_params(axis='y', which='both', left=True, right=False)

    plt.xlabel(x_label, position=(0.5, 5))
    plt.ylabel(y_label, position=(0, 0.5))

    plt.subplots_adjust(left=0.0 + right_shift)
    plt.title(plot_title)


    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig(f'plots/'+ plot_title.replace(" ", "_") + '_' + x_label + '_' + y_label, bbox_inches='tight')



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

def calculate_percentage_difference(base_value, candidate_value):

    if base_value == None or candidate_value == None:
        return None

    return (candidate_value - base_value) / base_value * 100


def tom(output_dictionary):

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

                y_label_values.append(model_name + '_' + optimization_label)

                print(base_prompt_filename, output_dictionary[code_problem][model_name][optimization_label]['filename'])


                similarity = calculate_similarity_between_files(base_prompt_filename, output_dictionary[code_problem][model_name][optimization_label]['filename'])
                energy = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'energy'][0]['mean'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'energy'][0]['mean'])
                memory = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'memory'][0]['mean'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'memory'][0]['mean'])
                flopst = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'flops'][0]['mean'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'flops'][0]['mean'])
                runtime = calculate_percentage_difference([x for x in base_prompt_statistics if x['type'] == 'runtime'][0]['mean'], [x for x in output_dictionary[code_problem][model_name][optimization_label]['statistics'] if x['type'] == 'runtime'][0]['mean'])

                data[y_label_values.index(model_name + '_' + optimization_label), x_label_values.index('similarity')] = similarity
                data[y_label_values.index(model_name + '_' + optimization_label), x_label_values.index('energy')] = energy
                data[y_label_values.index(model_name + '_' + optimization_label), x_label_values.index('memory')] = memory
                data[y_label_values.index(model_name + '_' + optimization_label), x_label_values.index('flops')] = flopst
                data[y_label_values.index(model_name + '_' + optimization_label), x_label_values.index('runtime')] = runtime

        plot(data, x_label_values, y_label_values, 'Metrics', 'Models', code_problem.replace('_', ' '), code_problem)



if __name__ == '__main__':
    dir = 'output_run0'
    output_filename='output_with_statistic.json'

    with open(dir + '/' + output_filename, 'r') as file:
        output_list_with_dictionaires = json.load(file)

    output_dictionary = process_output_list_with_dictionaries(output_list_with_dictionaires)

    tom(output_dictionary)




