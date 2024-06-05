from modules.interfaces.llms.adapter_abstract import LLMAdapter
import subprocess
import re


def parse_code_output(code_output: str) -> str:
    """
    Code is returned within the ''' ''' block. The first word is the
    programming language. This function returns a single string
    containing all code blocks concatenated with correct syntax.

    returns:
        a single string containing all code blocks with correct syntax
    """

    if '''```''' not in code_output:
        return code_output

    code_pattern = r"```(\w+)\s+(.*?)```"
    matches = re.findall(code_pattern, code_output, re.DOTALL)

    code_combined = ""
    for match in matches:
        code_combined += match[1].strip() + "\n\n"

    return code_combined.strip()


class DeepSeekCoder33bBase(LLMAdapter):

    def __init__(self):
        self.process_server = None
        self.process_client = None
        self.host = "localhost"
        self.port = 12345

    def initialize(self, server_is_listening=False) -> int:
        command = f"""ssh -tt tcappendij@129.152.15.84 'cd data/volume_2/Greencode_framework/ && source venv/bin/activate && python3 deepseek-coder-33b-base/server.py --port {self.port} --host {self.host}'"""
        self.process_server = subprocess.Popen([command],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)

        if self.process_server.stdout is not None:
            for line in self.process_server.stdout:
            #     # Here the initialization is done when the server is listening
            #     # The print statement inside the server of "Server is listening..."
            #     # is used as a signal to know when the server is ready. Because
            #     # the process_server never terminates, this is necessary.
                if "Server is listening..." in line.decode():
                    server_is_listening = True
                    break

                if "Address already in use" in line.decode():
                    print("Port is already in use. Trying another port.")
                    self.port -= 1
                    server_is_listening = self.initialize()

        # # Server is not initialized correctly
        if server_is_listening is False:
            return 0

        return 1

    def process_code_prompt(self, code_prompt: str) -> str:
        command = f"""ssh -tt tcappendij@129.152.15.84 'cd data/volume_2/Greencode_framework/ && source venv/bin/activate && python3 deepseek-coder-33b-base/client.py --prompt "{code_prompt}" --port {self.port} --host {self.host}'"""

        self.process_client = subprocess.Popen([command],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True)


        code_output = ""

        if self.process_client.stdout is not None:
            for line in self.process_client.stdout:
                code_output += line.decode()

        code_output = parse_code_output(code_output)

        return code_output

    def get_name(self) -> str:
        return "DeepSeek-Coder-33b-base"

    def close(self) -> int:
        print("Close DeepSeek-Coder-33b-base")

        self.process_code_prompt("exit")

        if self.process_server is not None:
            self.process_server.terminate()

        if self.process_client is not None:
            self.process_client.terminate()

        return 1
