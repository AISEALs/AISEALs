# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        n = 0
        i = head
        while i is not None:
            n += 1
            i = i.next
        s = []
        i = 0
        cur = head
        while i < n // 2:
            s.append(cur.val)
            cur = cur.next
            i += 1
        ans = 0
        while cur is not None:
            if cur.val + s[-1] > ans:
                ans = cur.val + s[-1]
            cur = cur.next
            s.pop(-1)
        return ans

    def pairSum(self, head: Optional[ListNode]) -> int:
        s = []
        cur = head
        while cur is not None:
            s.append(cur.val)
            cur = cur.next
        ans = 0
        while len(s) > 0:
            ans = max(ans, s.pop(0) + s.pop(-1))
        return ans

