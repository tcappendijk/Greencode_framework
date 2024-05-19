from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [prompt]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for result in results:
        print("Here is the code:")
        print({result['generation']})


if __name__ == "__main__":
    fire.Fire(main)