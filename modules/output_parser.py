import json

class OutputParser:
    """
        The input of this module is the output of the framework. This module is
        responsible for parsing the output of the framework and returning the
        filename where the output is written. The input of this module has the
        following structure:
        [{'model_name': , 'prompt': , 'filename': 'code': , statistics: [{'type': , 'name': , 'value': , 'unit': }]}]
    """

    def __init__(self, list_of_dicts: list) -> None:
        """
        Args:
            list_of_dicts (list): A list of dictionaries containing the model
            name, prompt, filename of the file that contains the code, the code,
            and the statistics.
        """
        self.__list_of_dicts = list_of_dicts


    def parse_output(self, output_file_path: str) -> str:
        """
        This method parses the output of the framework and writes it to a JSON file.

        Args:
            output_file_path (str): Path to the output JSON file.

        Returns:
            str: Message indicating the status of the operation.
        """
        try:
            with open(output_file_path, 'w') as f:
                json.dump(self.__list_of_dicts, f, indent=4)
            return f"Output successfully written to {output_file_path}"
        except Exception as e:
            return f"Error writing output to {output_file_path}: {str(e)}"