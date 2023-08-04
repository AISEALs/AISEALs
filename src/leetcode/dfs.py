from typing import List


def DFS(A):
    def backtrack(selects: List[int], path: List[int]):
        if len(path) == 0:  # is finish
            ans.append(path[:])
            return

        for i in range(len(selects)):
            path.append(selects[i])
            backtrack(path, selects[:i] + selects[i+1:])
            path.pop()

    ans = []
    backtrack(A, [])
    return ans

print(DFS([1, 2, 3]))
