
# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        if not head or not head.next:
            return head

        left = head
        right = self.getMid(head)
        tmp = right.next
        right.next = None
        right = tmp

        left = self.sortList(left)
        right = self.sortList(right)
        return self.merge(left, right)

    def getMid(self, head):
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def merge(self, list1, list2):
        tail = dummy = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        if list1:
            tail.next = list1
        if list2:
            tail.next = list2

        return dummy.next

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












































































































































































































































































































































































































































































































