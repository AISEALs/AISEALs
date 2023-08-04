# 请你设计一个数据结构，支持 添加新单词 和 查找字符串是否与任何先前添加的字符串匹配 。 
# 
#  实现词典类 WordDictionary ： 
# 
#  
#  WordDictionary() 初始化词典对象 
#  void addWord(word) 将 word 添加到数据结构中，之后可以对它进行匹配 
#  bool search(word) 如果数据结构中存在字符串与 word 匹配，则返回 true ；否则，返回 false 。word 中可能包含一些 
# '.' ，每个 . 都可以表示任何一个字母。 
#  
# 
#  
# 
#  示例： 
# 
#  
# 输入：
# ["WordDictionary","addWord","addWord","addWord","search","search","search",
# "search"]
# [[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
# 输出：
# [null,null,null,null,false,true,true,true]
# 
# 解释：
# WordDictionary wordDictionary = new WordDictionary();
# wordDictionary.addWord("bad");
# wordDictionary.addWord("dad");
# wordDictionary.addWord("mad");
# wordDictionary.search("pad"); // return False
# wordDictionary.search("bad"); // return True
# wordDictionary.search(".ad"); // return True
# wordDictionary.search("b.."); // return True
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= word.length <= 500 
#  addWord 中的 word 由小写英文字母组成 
#  search 中的 word 由 '.' 或小写英文字母组成 
#  最多调用 50000 次 addWord 和 search 
#  
#  Related Topics 深度优先搜索 设计 字典树 字符串 👍 392 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
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


class WordDictionary:
    def __init__(self):
        self.trie = Trie()

    def addWord(self, word: str) -> None:
        self.trie.insert(word)

    def search(self, word: str) -> bool:
        return self.trie.search(word)


# Your WordDictionary object will be instantiated and called as such:
if __name__ == '__main__':
    methods = ["WordDictionary", "addWord", "addWord", "addWord", "addWord", "search", "search", "addWord", "search", "search",
     "search", "search", "search", "search"]
    values = [[], ["at"], ["and"], ["an"], ["add"], ["a"], [".at"], ["bat"], [".at"], ["an."], ["a.d."], ["b."], ["a.d"], ["."]]

    wordDictionary = WordDictionary()
    for m, v in zip(methods, values):
        if m == 'addWord':
            wordDictionary.addWord(v[0])
        elif m == 'search':
            print(wordDictionary.search(v[0]))

#     [null,null,null,null,null,false,false,null,true,true,false,false,true,false]

# leetcode submit region end(Prohibit modification and deletion)
