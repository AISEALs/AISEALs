import heapq

class Node(object):
    def __init__(self, val: int, pos=(0, 0)):
        self.val = val
        self.pos = (val - pos[0], val - pos[1])

    def __repr__(self):
        return f'Node value: {self.val}'

    def __lt__(self, other):
        return self.val < other.val

heap = [Node(2), Node(0), Node(1), Node(4), Node(2)]
heapq.heapify(heap)
print(heap)  # output: [Node value: 0, Node value: 2, Node value: 1, Node value: 4, Node value: 2]

heapq.heappop(heap)
print(heap)  # output: [Node value: 1, Node value: 2, Node value: 2, Node value: 4]
