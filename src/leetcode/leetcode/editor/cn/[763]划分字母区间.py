# 字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。 
# 
#  
# 
#  示例： 
# 
#  
# 输入：S = "ababcbacadefegdehijhklij"
# 输出：[9,7,8]
# 解释：
# 划分结果为 "ababcbaca", "defegde", "hijhklij"。
# 每个字母最多出现在一个片段中。
# 像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
#  
# 
#  
# 
#  提示： 
# 
#  
#  S的长度在[1, 500]之间。 
#  S只包含小写字母 'a' 到 'z' 。 
#  
#  Related Topics 贪心 哈希表 双指针 字符串 👍 630 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import Counter
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        cnt = Counter()
        for i in s:
            cnt[i] += 1
        uniq_words = set()
        total_count = 0
        ans = []
        num = 0
        for i in s:
            if i not in uniq_words:
                total_count += cnt[i]
                uniq_words.add(i)
            total_count -= 1
            num += 1
            if total_count == 0:
                ans.append(num)
                uniq_words.clear()
                num = 0
        return ans


if __name__ == '__main__':
    print(Solution().partitionLabels("ababcbacadefegdehijhklij"))

# leetcode submit region end(Prohibit modification and deletion)
