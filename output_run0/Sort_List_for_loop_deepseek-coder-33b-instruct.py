
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

        mid = self.getMid(head)
        left = self.sortList(head)
        right = self.sortList(mid)

        return self.merge(left, right)

    def getMid(self, head):
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        mid = slow.next
        slow.next = None
        return mid

    def merge(self, head1, head2):
        dummy = ListNode(0)
        tail = dummy
        while head1 and head2:
            if head1.val < head2.val:
                tail.next = head1
                head1 = head1.next
            else:
                tail.next = head2
                head2 = head2.next
            tail = tail.next
        tail.next = head1 if head1 else head2
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


















































































































































































































































































































































































































































































































