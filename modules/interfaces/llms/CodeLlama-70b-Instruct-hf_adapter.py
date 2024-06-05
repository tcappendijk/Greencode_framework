from modules.interfaces.llms.adapter_abstract import LLMAdapter
import subprocess


class CodeLlama70bInstructhf(LLMAdapter):

    def __init__(self):
        self.handle_code_process = None

    def initialize(self) -> int:
        return 1

    def process_code_prompt(self, code_prompt: str) -> str:
        command = f"""ssh -tt tcappendij@129.152.15.84 'cd data/volume_2/Greencode_framework/ && source venv/bin/activate && torchrun --nproc_per_node 8 CodeLlama/handle_prompt_instruction.py      --ckpt_dir ../codellama/CodeLlama-70b/      --tokenizer_path ../codellama/CodeLlama-70b/tokenizer.model     --max_batch_size 1 --prompt "{code_prompt}"'"""

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
        return "CodeLlama-70b-Instruct-hf"

    def close(self) -> int:
        return 1
