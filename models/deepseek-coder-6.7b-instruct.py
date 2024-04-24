from models.model_abstract import Model
import subprocess


class DeepSeekCoder67bInstruct(Model):

    def __init__(self):
        self.process_server = None
        self.process_client = None

    def initialize(self) -> int:
        self.process_server = subprocess.Popen(["ssh -tt ssh://tcappendij@130.61.19.89 python3 project/Greencode_framework/deepseek-coder-6.7b-instruct/server.py" ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)

        if self.process_server.poll() is None:
            for line in self.process_server.stdout:
                print(line.decode(), end="")
                if "Server is listening..." in line.decode():
                    break

        print("DeepSeekCoder 6.7b Instruct initialized.")
        return 0

    def handle_code_prompt(self, code_prompt: str) -> str:
        self.process_client = subprocess.Popen(["ssh -tt ssh://tcappendij@130.61.19.89 python3 project/Greencode_framework/deepseek-coder-6.7b-instruct/client.py --prompt \\\"" + code_prompt + "\\\""],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True)


        code_output = ""

        if self.process_client.poll() is None:
            for line in self.process_client.stdout:
                code_output += line.decode()

        return code_output

    def get_name(self) -> str:
        return "DeepSeekCoder 6.7b Instruct"

    def close(self) -> int:
        print("Close DeepSeekCoder 6.7b Instruct.")

        self.handle_code_prompt("exit")

        for line in self.process_server.stdout:
            print(line.decode(), end="")

        if self.process_server is not None:
            self.process_server.terminate()

        if self.process_client is not None:
            self.process_client.terminate()

        return 0
