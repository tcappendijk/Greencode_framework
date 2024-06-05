import json

class InputProcessor:
    """
        A class that processes the input json file. The input json file contains
        the mode of the framework and the prompts for the framework. The
        prompts are stored in a list inside the json file.
    """

    def __init__(self, input_file):
        self.__mode = ""
        self.__prompts = []
        self.__input_file = input_file
        self.__prompt_labels = []

    def process_input(self):
        """
        Processes the input json file.
        """
        with open(self.__input_file, "r") as f:
            input_json = json.load(f)
            self.__mode = input_json["mode"]
            self.__prompts = input_json["prompts"]
            self.__prompt_labels = input_json["prompt_labels"]

    def get_mode(self):
        return self.__mode

    def get_prompts(self):
        return self.__prompts

    def get_prompt_labels(self):
        return self.__prompt_labels
