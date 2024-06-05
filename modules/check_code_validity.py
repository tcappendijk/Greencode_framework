class CheckCodeValidity:

    def __init__(self, list_of_dicts:list) -> None:
        """
        This class is responsible for checking the validity of the code by the user.

        Args:
            list_of_dicts (list): A list of dictionaries containing the model name, prompt and the
            output of the model.
        """
        self.__list_of_dicts = list_of_dicts

    def check_code_validity(self):
        """
        This method checks the validity of the code prodcued by the LLMS models.
        The code is written to a file and the user must add tests to check the
        validity of the code.

        Returns:
            list: A list of dictionaries containing the model name, prompt,
            filname of the file that contains the code, and the code.
        """

        print("--------------------------------------------------------------")
        print("Checking the validity of the code outputted by the LLMS models.")
        print("The user also must add tests to check the validity of the code.")
        print("Use the same tests for the same prompt in order to compare the outputs of the models.")
        print("--------------------------------------------------------------")

        checked_list_of_dicts = []

        for model_dict in self.__list_of_dicts:
            filename = model_dict['filename']
            print(f"Model Name: {model_dict['model_name']}")
            print(f"Prompt: {model_dict['prompt']}")
            print(f"The code is written to the file: " + filename)
            print("--------------------------------------------------------------")

            user_input = input("If the code is incorrect, type n else continue by pressing enter: ")

            if user_input == "n":
                continue

            with open(filename, "r") as f:
                code = f.read()
                f.close()

            model_dict['code'] = code
            checked_list_of_dicts.append(model_dict)

        return checked_list_of_dicts


