# 请你设计并实现一个满足 LRU (最近最少使用) 缓存 约束的数据结构。 
# 
#  实现 LRUCache 类： 
# 
#  
#  
#  
#  LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存 
#  int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。 
#  void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 
# key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。 
#  
# 
#  函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。 
#  
#  
# 
#  
# 
#  示例： 
# 
#  
# 输入
# ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
# [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
# 输出
# [null, null, null, 1, null, -1, null, -1, 3, 4]
# 
# 解释
# LRUCache lRUCache = new LRUCache(2);
# lRUCache.put(1, 1); // 缓存是 {1=1}
# lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
# lRUCache.get(1);    // 返回 1
# lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
# lRUCache.get(2);    // 返回 -1 (未找到)
# lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
# lRUCache.get(1);    // 返回 -1 (未找到)
# lRUCache.get(3);    // 返回 3
# lRUCache.get(4);    // 返回 4
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= capacity <= 3000 
#  0 <= key <= 10000 
#  0 <= value <= 10⁵ 
#  最多调用 2 * 10⁵ 次 get 和 put 
#  
#  Related Topics 设计 哈希表 链表 双向链表 👍 1807 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Node:
    def __init__(self, key=None, value=None, pre=None, next=None):
        self.key = key
        self.value = value
        self.pre = pre
        self.next = next

class LRUCache:
    def __init__(self, capacity: int):
        self.m = dict()
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.pre = self.head
        self.cur_size = 0
        self.capacity = capacity

    def get(self, key: int) -> int:
        rs = -1
        if key in self.m:
            node = self.m.get(key)
            self._move_to_head(node)
            rs = node.value
        self.assert_len()
        return rs

    def put(self, key: int, value: int) -> None:
        if key in self.m:
            node = self.m.get(key)
            node.value = value
            self._move_to_head(node)
        else:
            node = Node(key, value)
            self.m[key] = node
            self._add_to_head(node)
        if self.cur_size > self.capacity:
            self.m.pop(self.tail.pre.key)
            self._delete_node(self.tail.pre)
        self.assert_len()

    def _move_to_head(self, node: Node):
        if self.head.next != node:
            self._delete_node(node)
            self._add_to_head(node)

    def _delete_node(self, node: Node):
        pre = node.pre
        next = node.next
        pre.next = next
        next.pre = pre
        self.cur_size -= 1

    def _add_to_head(self, node: Node):
        node.next = self.head.next
        self.head.next.pre = node
        self.head.next = node
        node.pre = self.head
        self.cur_size +=1

    def assert_len(self):
        pass
        # num = 0
        # datas = []
        # node = self.head.next
        # while node != self.tail:
        #     num += 1
        #     datas.append((node.key, node.value))
        #     node = node.next
        # assert num == self.cur_size
        # print(datas)

# Your LRUCache object will be instantiated and called as such:
if __name__ == '__main__':
    lRUCache = LRUCache(2)
    print(lRUCache.put(2, 1))
    print(lRUCache.put(2, 2))
    print(lRUCache.get(2))
    print(lRUCache.put(1, 1))
    print(lRUCache.put(4, 1))
    print(lRUCache.get(2))

# leetcode submit region end(Prohibit modification and deletion)
