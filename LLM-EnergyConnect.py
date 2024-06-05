from modules.input_processor import InputProcessor
from modules.process_code_and_mode_prompt import ProcessCodePrompt
from modules.check_code_validity import CheckCodeValidity
from modules.output_parser import OutputParser
from modules.measure_statistics import MeasureStatistics

import argparse

def main():
    parser = argparse.ArgumentParser(description="LLM Energy Connect")
    parser.add_argument("input_file", type=str, help="Input json file")
    args = parser.parse_args()

    input_parser_obj = InputProcessor(args.input_file)
    input_parser_obj.process_input()

    mode = input_parser_obj.get_mode()
    prompts = input_parser_obj.get_prompts()
    prompt_labels = input_parser_obj.get_prompt_labels()

    process_mode_and_code_prompt_obj = ProcessCodePrompt(mode, prompts, prompt_labels)
    outputs = process_mode_and_code_prompt_obj.process_code_prompts()

    if outputs == []:
        return

    # When the output of the LLMs is only code that does not need to be checked,
    # the check_code_validity object will can be skipped
    check_code_validity_obj = CheckCodeValidity(outputs)
    outputs = check_code_validity_obj.check_code_validity()

    measurement_tools_handler_obj = MeasureStatistics()
    measurement_tools_handler_obj.initialize_all_tools()

    checked_output = measurement_tools_handler_obj.measure_statistics(outputs)

    output_parser_obj = OutputParser(checked_output)
    print(output_parser_obj.parse_output('output.json'))

if __name__ == "__main__":
    main()