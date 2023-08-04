# 你写下了若干 正整数 ，并将它们连接成了一个字符串 num 。但是你忘记给这些数字之间加逗号了。你只记得这一列数字是 非递减 的且 没有 任何数字有前导 0
#  。 
# 
#  请你返回有多少种可能的 正整数数组 可以得到字符串 num 。由于答案可能很大，将结果对 10⁹ + 7 取余 后返回。 
# 
#  
# 
#  示例 1： 
# 
#  输入：num = "327"
# 输出：2
# 解释：以下为可能的方案：
# 3, 27
# 327
#  
# 
#  示例 2： 
# 
#  输入：num = "094"
# 输出：0
# 解释：不能有数字有前导 0 ，且所有数字均为正数。
#  
# 
#  示例 3： 
# 
#  输入：num = "0"
# 输出：0
# 解释：不能有数字有前导 0 ，且所有数字均为正数。
#  
# 
#  示例 4： 
# 
#  输入：num = "9999999999999"
# 输出：101
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= num.length <= 3500 
#  num 只含有数字 '0' 到 '9' 。 
#  
#  Related Topics 字符串 动态规划 后缀数组 👍 15 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from functools import lru_cache


class Solution:
    def numberOfCombinations(self, num: str) -> int:
        length = len(num)
        ans = 0
        def traceback(index: int, last: int):
            nonlocal ans
            if index == length:
                ans += 1
                return
            for i in range(index+1, length+1):
                s = num[index:i]
                if s.startswith('0'):
                    return
                if int(s) >= last:
                    traceback(i, int(s))

        traceback(0, 0)
        return ans

if __name__ == '__main__':
    s = Solution()
    print(s.numberOfCombinations('181599706296201533688444310698720506149731032417146774186256527047743490211586938068687937416089'))
    print(s.numberOfCombinations.cache_info())
# leetcode submit region end(Prohibit modification and deletion)
