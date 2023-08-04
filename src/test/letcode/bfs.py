from typing import List


class Node:
    def adj(self) -> List:
        pass

    pass


# 计算起点start到终点target的距离
def BFS(start: Node, target: Node):
    q = []
    visited = set()  # 记录走过的点

    q.append(start)
    visited.add(start)
    step = 0

    while len(q) > 0:
        sz = len(q)
        # 将当前队列的所有节点四周扩散
        for i in range(sz):
            cur = q[0]
            if cur == target:
                return step
            # 将cur的相邻节点加入队列
            for x in cur.adj():
                if x not in visited:
                    q.append(x)
                    visited.add(x)
        # 更新步数
        step += 1
    return step
