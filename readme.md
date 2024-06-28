# Greencode Framework

This is a framework that can handle multiple LLMs and measurement tools. The goal is that the user can input prompt(s), which are sent to the connected LLMs. The LLMs handle the prompts and return the results. In the current version of the framework, the code output is validated and corrected by the user. When the user has corrected the code output, the statistics of the code output are calculated and returned to the user in a JSON file. The main file of the framework is named `FMECG-LLM.py`.

# Installation
We created a `requirements.txt` file that contains all the necessary packages to run the framework. To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```
The libraries specific to the implementation of an LLM or measurement tool should be installed separately.

# Input
The input of the framework is a JSON file containing the mode, prompts, and prompt labels. The mode can either be sequential or parallel. The chosen mode depends on the user's access to the LLMs. If the LLMs connected by the user can handle multiple prompts at the same time, the mode should be set to parallel. If the LLMs can only handle one prompt at a time, the mode should be set to sequential. The prompts contain the code problem that the user wants to solve. The prompt labels are used to create the output files with a corresponding name. The input JSON file should look like this:
```json
{
    "mode": "sequential/parallel",
    "prompts": [
        "List of prompts to be sent to the LLMs."
    ],
    "prompt_labels": [
        "List of labels for the prompts."
    ]
}
```

# Output
The output of the framework is a JSON file that contains the following keys:
```json
{
    "model_name": "The name of the model used for generating the output.",
    "prompt": "The prompt that was given as input.",
    "code": "The code that was generated as output.",
    "prompt_label": "The label associated with the prompt.",
    "filename": "The filename of the output file.",
    "statistics": [
        {
            "type": "The type of statistic (e.g., memory, energy, runtime, flops).",
            "values": ["An array of values for the statistic."],
            "unit": "The unit of measurement for the statistic."
        }
    ]
}
```
Please note that the actual values for the statistics and model name are dependent on LLMs and measurement tools connected by the user.

# Connecting LLMs

The user can connect multiple LLMs to the framework. To connect an LLM, the user must implement an adapter abstract located at:

```bash
modules/interfaces/llms/adapter_abstract.py
```

In the same folder, there are examples of implemented adapters for different LLMs. To use the framework, the user either must implement the LLMs to use these examples or delete the examples, otherwise they would generate errors. The functions in `adapter_abstract.py` all must be implemented to connect a new LLM.

These functions are:

- `initialize`: Function to initialize the model. Returns 1 if successful, 0 otherwise.
- `process_code_prompt`: Function to process the code prompt by the LLM. Returns the generated code or an empty string if the model cannot not handle the code prompt.
- `get_name`: Function to get the name of the model. The name should match the name of the file that the model is implemented in. Returns the name of the model.
- `close`: Function to close the model. Returns 1 if successful, 0 otherwise.

We refer to the `modules/interfaces/llms/adapter_abstract.py` file for more details on implementing these functions. The new implementation of the user must be placed in the `modules/interfaces/llms/` folder.

# Connecting Measurement Tools
The user can connect multiple measurement tools to the framework. To connect a measurement tool, the user must implement an adapter abstract located at:

```bash
modules/interfaces/measurement_tools/adapter_abstract.py
```

In the same folder, there are examples of implemented adapters for different measurement tools. These measurement tools require specific hardware and an installation and setup to work. So these are just examples and should be replaced by the user's implementation, or the user must implement them thereself. The functions in adapter_abstract.py must all be implemented to connect a new measurement tool.

These functions are:
The measurement tool adapter abstract should include the following functions:

- `initialize`: Function to initialize the measurement tool. Returns 1 if successful, 0 otherwise.
- `measure_statistics`: Function to measure the statistic of the code. Receives the command to measure the statistic of and returns a list with one or multiple dictionaries. The keys of the dictionary are 'type', 'values', 'unit'. The type is the type of the statistic, the value is the value of the statistic, and the unit is the unit of the statistic. On error, the function should return an empty dictionary. Multiple dictionaries can be returned if multiple statistics are measured. On failure, the value of the statistic should be None.
- `get_name`: Function to get the name of the measurement tool. Returns the name of the measurement tool.

We refer to the `modules/interfaces/measurement_tools/adapter_abstract.py` file for more details on implementing these functions. The new implementation of the user must be placed in the `modules/interfaces/measurement_tools/` folder

