"""
    A file that generates the input json file for the program. The input of the
    program is the mode and the prompts.
"""

import json
import argparse
import os
import copy

def generate_input_json(mode, prompts_dir, opt_sent_file):
    """
    Generates the input json file for the program.
    Args:
        mode: The mode of the framework.
        prompts: The prompts for the framework.
    """

    prompts = []
    prompt_labels = []


    for filename in os.listdir(prompts_dir):
        if 'optimization' in filename:
            continue

        with open(os.path.join(prompts_dir, filename), "r") as f:
            prompt = f.read()

            prompt = prompt.replace("\"", "\\\"")

            prompts.append(prompt)
            label, _ = os.path.splitext(filename)
            prompt_labels.append(label)
            f.close()

    with open(opt_sent_file, "r") as f:
        cache_prompts = copy.deepcopy(prompts)
        for line in f:
            label = None
            if 'energy' in line:
                label = 'energy'
            elif 'library functions' in line:
                label = 'library_functions'
            elif 'for loop' in line:
                label = 'for_loop'

            for (index, prompt) in enumerate(prompts):
                cache_prompts.append(line + '\n' + prompt)
                prompt_labels.append(prompt_labels[index] + '_' + label)

        prompts = cache_prompts

    input_json = {
        "mode": mode,
        "prompts": prompts,
        "prompt_labels": prompt_labels
    }

    with open("input.json", "w") as f:
        json.dump(input_json, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="The mode of the framework.", default='sequential')
    parser.add_argument("--prompts_dir", help="The directory of the prompts.", default='input_prompts')
    parser.add_argument("--opt_sent_file", help="The file containing the optimization sentences.", default='input_prompts/prompt_optimize_sentences.txt')
    args = parser.parse_args()

    generate_input_json(args.mode, args.prompts_dir, args.opt_sent_file)