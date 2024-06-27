
# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
        nums.sort()
        head = ListNode(nums[0])
        node = head
        for i in range(1, len(nums)):
            node.next = ListNode(nums[i])
            node = node.next
        return head


def create_linked_list(vals):
    dummy = ListNode()
    current = dummy
    for val in vals:
        current.next = ListNode(val)
        current = current.next
    return dummy.next

def linked_list_to_list(head):
    vals = []
    current = head
    while current:
        vals.append(current.val)
        current = current.next

    return vals

for _ in range(40):

    input_list = [i for i in range(5 * (10 ** 4), 0, -1)]
    output_list = sorted(input_list)

    head = create_linked_list(input_list)

    solution = Solution()
    sorted_head = solution.sortList(head)

    solution_list = linked_list_to_list(sorted_head)

    assert solution_list == output_list