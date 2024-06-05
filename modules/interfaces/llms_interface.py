from modules.interfaces.llms.adapter_abstract import LLMAdapter

import os
import threading
import importlib.util

class LLMsInterface:
    def __init__(self, llm_dir: str = "modules/interfaces/llms"):
        self.models = []

        if not os.path.exists(llm_dir):
            raise FileNotFoundError("Directory with the large language models does not exist.")

        for file_name in os.listdir(llm_dir):
            if file_name.endswith(".py") and file_name != "__init__.py":
                module_name = os.path.splitext(file_name)[0]
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(llm_dir, file_name))

                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, LLMAdapter) and obj is not LLMAdapter:
                        try:
                            obj()
                        except Exception as e:
                            print(f"Error while initializing model {obj.__name__}: {e}")
                            continue

                        self.models.append(obj())

    def sequential_process_code_prompt(self, code_prompts: list, prompt_labels: list):
        """
        This function makes it possible to process a list of code prompts
        for each model. This function iterated model for model and does the
        following: initializes the model, procceses the code prompts and closes
        the model. The output of the models is returned. The output is a list
        of dictionaries containing the model name, prompt and the output of the
        model.

        Args:
            code_prompts (list): A list of code prompts.

        Returns:
            list: A list of dictionaries containing the model name, prompt and the
            code output of the model.
        """
        models_output = []

        models_with_initilization_error = []
        for model in self.models:
            model_name = model.get_name()
            if model.initialize() == 0:
                print(f"Error while initializing model {model_name}")
                models_with_initilization_error.append(model)
            else:
                for index, prompt in enumerate(code_prompts):
                    code = model.process_code_prompt(prompt)
                    models_output.append({"model_name": model_name, "prompt": prompt, "code": code, "prompt_label": prompt_labels[index]})

                model.close()

        for model in models_with_initilization_error:
            self.models.remove(model)

        return models_output

    def parallel_process_code_prompt(self, code_prompts: list, prompt_labels: list):
        """
        This method makes it possible to process a list of code prompts
        for each model in parallel. This function initializes all the models
        in parallel and then processes the code prompts for each model in parallel.
        The output is a list of dictionaries containing the model name, prompt
        and the output of the model.

        Args:
            code_prompts (list): List of code prompts to send to the models.

        Returns:
            list: A list of dictionaries containing the model name, prompt and the
            code output of the model.
        """

        def initialize_model(model):
            if model.initialize() == 0:
                print(f"Error while initializing model {model.get_name()}")
                # Delete the model from the list of models so that it is not used but the other models are still used.
                self.models.remove(model)

        def process_code_prompts(model, code_prompts, models_output, prompt_labels):
            for index, prompt in enumerate(code_prompts):
                code = model.process_code_prompt(prompt)
                models_output.append({"model_name": model.get_name(), "prompt": prompt, "code": code, "prompt_label": prompt_labels[index]})


            model.close()


        models_output = []
        threads = []

        for model in self.models:
            t = threading.Thread(target=initialize_model, args=(model,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        threads = []
        for model in self.models:
            t = threading.Thread(target=process_code_prompts, args=(model, code_prompts, models_output, prompt_labels))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return models_output
