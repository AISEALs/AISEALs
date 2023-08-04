# 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。 
# 
#  给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。 
# 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：digits = "23"
# 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
#  
# 
#  示例 2： 
# 
#  
# 输入：digits = ""
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：digits = "2"
# 输出：["a","b","c"]
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= digits.length <= 4 
#  digits[i] 是范围 ['2', '9'] 的一个数字。 
#  
#  Related Topics 哈希表 字符串 回溯 👍 1680 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
        az = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        d = dict()
        start = 0
        for i, num in enumerate([3, 3, 3, 3, 3, 4, 3, 4]):
            d[i+2] = az[start: start+num]
            start += num
        ans = []

        def backtrack(s: List[str]):
            if len(digits) == len(s):
                ans.append(''.join(s))
                return
            digit = int(digits[len(s)])
            for c in d[digit]:
                s.append(c)
                backtrack(s)
                s.pop(-1)

        backtrack([])
        return ans


if __name__ == '__main__':
    print(Solution().letterCombinations(''))
# leetcode submit region end(Prohibit modification and deletion)
