from models.model_abstract import Model
import subprocess


class DeepSeekCoder67bInstruct(Model):
    def initialize(self) -> int:

        # Execute the server of the model inside a subprocess
        process = subprocess.Popen(["ssh ssh://tcappendij@130.61.19.89 && cd project/Greencode_framework/ && python3 deepseek-coder-6.7b-instruct/server.py"], stdout=subprocess.PIPE)

        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.strip().decode("utf-8"))

        print("DeepSeekCoder 6.7b Instruct initialized.")
        return 0

    def handle_code_prompt(self, code_prompt: str) -> str:
        print("DeepSeekCoder 6.7b Instruct handling code prompt.")
        return "DeepSeekCoder 6.7b Instruct generated code."

    def get_name(self) -> str:
        return "DeepSeekCoder 6.7b Instruct"

    def close(self) -> int:
        print("DeepSeekCoder 6.7b Instruct closed.")
        return 0
