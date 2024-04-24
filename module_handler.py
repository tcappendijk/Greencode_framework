from models.model_abstract import Model

import os
import threading
import importlib.util

class ModelHandler:
    def __init__(self):
        self.models = []

    def load_models(self):
        models_dir = "models"
        if not os.path.exists(models_dir):
            raise FileNotFoundError("Models directory not found.")

        for file_name in os.listdir(models_dir):
            if file_name.endswith(".py") and file_name != "__init__.py":
                module_name = os.path.splitext(file_name)[0]
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(models_dir, file_name))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
                        try:
                            obj()
                        except Exception as e:
                            print(f"Error while initializing model {obj.__name__}: {e}")
                            continue

                        self.models.append(obj())

    def get_models(self):
        return self.models

    def initialize_models(self):

        def initialize_model(model):
            model.initialize()

        threads = []
        for model in self.models:
            thread = threading.Thread(target=initialize_model, args=(model,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def handle_code_prompt(self, code_prompt):
        code_output_models = []
        for model in self.models:
            model_name = model.get_name()
            code_output = model.handle_code_prompt(code_prompt)
            code_output_models.append((model_name, code_output))
        return code_output_models

    def close_models(self):
        for model in self.models:
            model.close()


# For testing now
if __name__ == "__main__":
    handler = ModelHandler()
    handler.load_models()

    handler.initialize_models()

    while True:
        code_prompt = input("Enter code prompt: ")

        if code_prompt == "exit":
            handler.close_models()
            break

        print(handler.handle_code_prompt(code_prompt)[0][1])