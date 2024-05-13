from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

custom_cache_dir = "/data/volume_2"
token = "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"
model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")

code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

# Generate code for an input string
input_string = """Given the head of a linked list, return the list after sorting it in ascending order.
Example 1:

Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:

Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Example 3:

Input: head = []
Output: []

Constraints:

    The number of nodes in the list is in the range [0, 5 * 104].
    -105 <= Node.val <= 105

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
"""
# input_string = "Given the head of a linked list, return the list after sorting it in ascending order.    Example 1:  Input: head = [4,2,1,3] Output: [1,2,3,4]  Example 2:  Input: head = [-1,5,3,4,0] Output: [-1,0,3,4,5]  Example 3:  Input: head = [] Output: []    Constraints:  The number of nodes in the list is in the range [0, 5 * 104]. -105 <= Node.val <= 105    Follow up: Can you sort the linked list in O(n logn) time and O(1) memory (i.e. constant space)?  # Definition for singly-linked list. # class ListNode: #     def __init__(self, val=0, next=None): #         self.val = val #         self.next = next class Solution: def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]: "


generated_code = code_generator(input_string, max_length=1000)[0]['generated_text']
print(generated_code)

