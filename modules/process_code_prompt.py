from modules.interfaces.llms_interface import LLMsInterface
import os

class ProcessCodePrompt:

    def __init__(self, mode, prompts, prompt_labels) -> None:
        self.__mode = mode
        self.__prompts = prompts
        self.__prompts_labels = prompt_labels
        self.__llms_interface = LLMsInterface()

    def process_code_prompts(self, output_dir='output/'):
        """
        This function makes it possible to process a list of code prompts. This
        function communicates with the llms_interface to process the code prompts
        in parallel or sequentially. The large language model provides
        functions that can be used to process the code prompts sequentially or
        in parallel. The output is a list of dictionaries containing the model
        name, prompt and the output of the model.

        Returns:
            list: A list of dictionaries, where the dictionaries contain the
            keys 'model_name', 'prompt','prompt_label', and 'code'. If the mode
            is not valid, an empty list is returned.
        """
        output = None
        if self.__mode == "parallel":
            output = self.__llms_interface.parallel_process_code_prompt(self.__prompts, self.__prompts_labels)
        elif self.__mode == "sequential":
            output = self.__llms_interface.sequential_process_code_prompt(self.__prompts, self.__prompts_labels)
        else:
            print(f"Invalid mode {self.__mode}")
            return []

        if output is None:
            return []

        # Check if the ouput directory exists
        if output_dir[-1] != "/":
            output_dir += "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the output to a file, and add the filename to the dictionary
        for model_dict in output:
            filename = f"{model_dict['prompt_label']}_{model_dict['model_name']}.py"
            filename = output_dir + filename

            with open(filename, "w") as f:
                f.write(model_dict['code'])
                f.close()

            model_dict['filename'] = filename

        return output