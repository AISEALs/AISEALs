from typing import Optional, List
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


stack = []


def traverse(root: Optional[TreeNode]) -> List[int]:
    visited = TreeNode()
    pushLeftBranch(root)
    while stack:
        p = stack[-1]
        if (p.left is None or p.left == visited) and p.right != visited:
            # todo: 中序遍历代码位置
            pushLeftBranch(p.right)  # 遍历p的右子树
        if p.right is None or p.right == visited:   # p的右子树已被遍历
            # todo: 后序遍历代码位置
            print(p.val, end=' ')
            visited = stack.pop(-1)


def pushLeftBranch(root):
    while root:
        stack.append(root)
        root = root.left


def createBinaryTree(data) -> Optional[TreeNode]:
    if not data:
        return None
    n = iter(data)
    tree = TreeNode(next(n))
    fringe = deque([tree])
    while fringe:
        head = fringe.popleft()
        try:
            head.left = TreeNode(next(n))
            fringe.append(head.left)
            head.right = TreeNode(next(n))
            fringe.append(head.right)
        except StopIteration:
            break
    return tree


if __name__ == '__main__':
    data = [5,3,6,2,4,None,None,1]
    root = createBinaryTree(data)
    traverse(root)

