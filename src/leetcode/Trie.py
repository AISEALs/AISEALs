class Node:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.children = dict()


def _getNode(node, key):
    cur = node
    if cur is None:
        return False
    for c in key:
        if c not in cur.children.keys():
            return None
        cur = cur.children[c]
    return cur


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children.keys():
                cur.children[c] = Node()
            cur = cur.children[c]
        cur.leaf = True

    def get(self, word):
        return _getNode(self.root, word)

    def search(self, word: str) -> bool:
        return self._search(self.root, word)

    def _search(self, node, word) -> bool:
        if len(word) == 0:
            return node.leaf
        cur = node
        c = word[0]
        if c == '.':
            for k, v in cur.children.items():
                if self._search(v, word[1:]):
                    return True
            return False
        else:
            if c in cur.children.keys():
                return self._search(cur.children[c], word[1:])
            else:
                return False

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children.keys():
                return False
            cur = cur.children[c]
        return True
