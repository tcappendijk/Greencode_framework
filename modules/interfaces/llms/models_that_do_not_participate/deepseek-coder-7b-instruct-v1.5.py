from modules.handlers.llms.model_abstract import LLM
import subprocess
import re


class  DeepSeekCoder7bInstructv15(LLM):

    def __init__(self):
        self.handle_code_process = None

    def initialize(self) -> int:
        return 1

    def handle_code_prompt(self, code_prompt: str) -> str:
        model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
        command = f"""ssh -tt tcappendij@129.152.15.84 'cd data/volume_2/Greencode_framework/ && source venv/bin/activate && python3 huggingface_model/handle_code_prompt.py --model_name {model_name} --prompt "{code_prompt}"'"""

        self.handle_code_process = subprocess.Popen([command],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True)

        code_output = ""
        code_start = False
        if self.handle_code_process.stdout is not None:
            for line in self.handle_code_process.stdout:
                if "Here is the code:" in line.decode():
                    code_start = True
                elif code_start:
                    code_output += line.decode()

        return code_output

    def get_name(self) -> str:
        return "deepseek-coder-7b-instruct-v1.5"

    def close(self) -> int:
        return 1
