def DFS(A):
    result = []

    def backtrack(path=[], selects=[]):
        if path is finish:
            result.append(path[:])
            return

        for s in selects:
            path.append(s)
            backtrack(path, selects[:].remove(s))
            path.pop()

    backtrack([], A)
    return result


print(DFS([1, 2, 3]))
