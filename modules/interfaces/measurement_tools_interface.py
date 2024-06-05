from modules.interfaces.measurement_tools.adapter_abstract import MeasurementToolAdapter

import os
import importlib.util

class MeasurementInterface:
    def __init__(self, llm_dir: str = "modules/interfaces/measurement_tools"):
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
                    if isinstance(obj, type) and issubclass(obj, MeasurementToolAdapter) and obj is not MeasurementToolAdapter:
                        try:
                            obj()
                        except Exception as e:
                            print(f"Error while initializing model {obj.__name__}: {e}")
                            continue

                        self.models.append(obj())

    def initialize_all_tools(self):
        """
        This function initializes all the measurement tools. On succes, it returns 1,
        otherwise 0.
        """
        for model in self.models:
            model_name = model.get_name()
            if model.initialize() == 0:
                print(f"Error while initializing model {model_name}")
                return 0

        return 1

    def measure_statistics(self, command: str):
        """
        This function measures the statistic of the code using all the measurement tools.
        Args:
            command: The command to measure the statistic of.
        Returns:
            list: A list of dictionaries containing the type, value and
            unit of the statistic. If the measurement failed, it will not be
            included in the list.
        """

        models_output = []
        for model in self.models:
            statistics = model.measure_statistics(command)

            for statistic_dict in statistics:
                if statistic_dict != {}:
                    models_output.append(statistic_dict)

        return models_output
