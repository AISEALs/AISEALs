# 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：head = [1,2,3,4]
# 输出：[2,1,4,3]
#  
# 
#  示例 2： 
# 
#  
# 输入：head = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：head = [1]
# 输出：[1]
#  
# 
#  
# 
#  提示： 
# 
#  
#  链表中节点的数目在范围 [0, 100] 内 
#  0 <= Node.val <= 100 
#  
#  Related Topics 递归 链表 👍 1203 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dumy = ListNode(0, head)
        cur = dumy
        i = cur.next
        if not i:
            return i
        j = i.next
        while i and j:
            cur.next = j
            i.next = j.next
            j.next = i
            cur = i
            i = i.next
            if i:
                j = i.next
        return dumy.next

def printList(head):
    l = []
    cur = head
    while cur:
        l.append(cur.val)
        cur = cur.next
    print(l)


if __name__ == '__main__':
    nums = [1, 2, 3, 4, 5]
    node = None
    for i in reversed(nums):
        node = ListNode(i, node)
    head = node
    printList(head)
    ret = Solution().swapPairs(head)
    printList(ret)



# leetcode submit region end(Prohibit modification and deletion)
