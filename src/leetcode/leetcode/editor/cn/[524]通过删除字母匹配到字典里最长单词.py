# 给你一个字符串 s 和一个字符串数组 dictionary ，找出并返回 dictionary 中最长的字符串，该字符串可以通过删除 s 中的某些字符得到。
#  
# 
#  如果答案不止一个，返回长度最长且字母序最小的字符串。如果答案不存在，则返回空字符串。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
# 输出："apple"
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "abpcplea", dictionary = ["a","b","c"]
# 输出："a"
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 1000 
#  1 <= dictionary.length <= 1000 
#  1 <= dictionary[i].length <= 1000 
#  s 和 dictionary[i] 仅由小写英文字母组成 
#  
#  Related Topics 数组 双指针 字符串 排序 👍 282 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List

class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        # t可以通过s删除某些字符得到
        def satisfy(s, t) -> bool:
            i = j = 0
            while i < len(t) and j < len(s):
                if t[i] == s[j]:
                    i += 1
                j += 1
            return i >= len(t)

            # while i < len(t):
            #     while j < len(s):
            #         if t[i] == s[j]:
            #             i += 1
            #             j += 1
            #             break
            #         else:
            #             j += 1
            #     if i < len(t) and j >= len(s):
            #         return False
            #
            # return i >= len(t)

        ans = ''
        for t in dictionary:
            if satisfy(s, t):
                if len(t) > len(ans):
                    ans = t
                elif len(t) == len(ans):
                    i = 0
                    while i < len(ans) and t[i] == ans[i]:
                        i += 1
                    if i < len(ans) and t[i] < ans[i]:
                        ans = t
        return ans

if __name__ == '__main__':
    print(Solution().findLongestWord("wordgoodgoodgoodbestword", ["word","good","best","good"]))

# leetcode submit region end(Prohibit modification and deletion)
