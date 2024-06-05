from modules.input_processor import InputProcessor
from modules.process_code_and_mode_prompt import ProcessCodePrompt
from modules.check_code_validity import CheckCodeValidity
from modules.output_parser import OutputParser
from modules.measure_statistics import MeasureStatistics

import json
import argparse

def main():
    # output_dirs = ['output_run0', 'output_run1', 'output_run2', 'output_run3', 'output_run4', 'output_run5', 'output_run6', 'output_run7', 'output_run8', 'output_run9']
    output_dirs = ['output_run0']
    parser = argparse.ArgumentParser(description="LLM Energy Connect")
    parser.add_argument("input_file", type=str, help="Input json file")
    args = parser.parse_args()

    # input_parser_obj = InputProcessor(args.input_file)
    # input_parser_obj.process_input()

    # mode = input_parser_obj.get_mode()
    # prompts = input_parser_obj.get_prompts()
    # prompt_labels = input_parser_obj.get_prompt_labels()

    # process_mode_and_code_prompt_obj = ProcessCodePrompt(mode, prompts, prompt_labels)
    # outputs = process_mode_and_code_prompt_obj.process_code_prompts(output_dir)

    # for output_dir in reversed(output_dirs):
    #     print(output_dir)
    #     input_parser_obj = InputProcessor(args.input_file)
    #     input_parser_obj.process_input()

    #     mode = input_parser_obj.get_mode()
    #     prompts = input_parser_obj.get_prompts()
    #     prompt_labels = input_parser_obj.get_prompt_labels()

    #     process_mode_and_code_prompt_obj = ProcessCodePrompt(mode, prompts, prompt_labels)
    #     outputs = process_mode_and_code_prompt_obj.process_code_prompts(output_dir)

    #     with open(output_dir + '/output.json', 'w') as f:
    #         json.dump(outputs, f)


    measure_statistics_obj = MeasureStatistics()
    if measure_statistics_obj.initialize_all_tools() == 0:
        print("Error while initializing the measurement tools")
        return

    for output_dir in output_dirs:
        with open(output_dir + '/output.json', 'r') as f:
            outputs = json.load(f)

        outputs = measure_statistics_obj.measure_statistics(outputs)
        output_parser_obj = OutputParser(outputs)
        print(output_parser_obj.parse_output(output_dir + '/output_with_statistic.json'))



    # if outputs == []:
    #     return


    # for output_dir in output_dirs:
    #     with open(output_dir + '/output.json', 'r') as f:
    #         outputs = json.load(f)

    #     check_code_validity_obj = CheckCodeValidity(outputs)
    #     outputs = check_code_validity_obj.check_code_validity()


    #     measure_statistics_obj = MeasureStatistics(outputs)
    #     outputs = measure_statistics_obj.measure_statistics()


    #     output_parser_obj = OutputParser(outputs)
    #     print(output_parser_obj.parse_output(output_dir + 'output_with_statistic.json'))


    # # When the output of the LLMs is only code that does not need to be checked,
    # # the check_code_validity object will can be skipped
    # check_code_validity_obj = CheckCodeValidity(outputs)
    # outputs = check_code_validity_obj.check_code_validity()


    # measure_statistics_obj = MeasureStatistics(outputs)
    # outputs = measure_statistics_obj.measure_statistics()


    # output_parser_obj = OutputParser(outputs)
    # print(output_parser_obj.parse_output('output.json'))

if __name__ == "__main__":
    main()