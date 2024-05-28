from modules.input_parser import InputParser
from modules.handle_code_prompt import HandleCodePrompt
from modules.check_code_validity import CheckCodeValidity
from modules.handlers.measurement_tools_handler import MeasurementToolsHandler
from modules.output_parser import OutputParser

import argparse

def main():
    parser = argparse.ArgumentParser(description="LLM Energy Connect")
    parser.add_argument("input_file", type=str, help="Input json file")
    args = parser.parse_args()

    input_parser_obj = InputParser(args.input_file)
    input_parser_obj.parse_input()

    mode = input_parser_obj.get_mode()
    prompts = input_parser_obj.get_prompts()
    prompt_labels = input_parser_obj.get_prompt_labels()

    handle_mode_and_code_prompt_obj = HandleCodePrompt(mode, prompts, prompt_labels)
    outputs = handle_mode_and_code_prompt_obj.handle_code_prompts()

    if outputs == []:
        return

    # When the output of the LLMs is only code that does not need to be checked,
    # the check_code_validity object will can be skipped
    check_code_validity_obj = CheckCodeValidity(outputs)
    outputs = check_code_validity_obj.check_code_validity()

    measurement_tools_handler_obj = MeasurementToolsHandler()
    measurement_tools_handler_obj.initialize_all_tools()

    checked_output = []
    for output in outputs:
        output['statistics'] = measurement_tools_handler_obj.measure_all_statistic('python3 ' + output['filename'])
        checked_output.append(output)

    output_parser_obj = OutputParser(checked_output)
    print(output_parser_obj.parse_output('output.json'))

if __name__ == "__main__":
    main()