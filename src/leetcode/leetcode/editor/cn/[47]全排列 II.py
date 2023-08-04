# 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,1,2]
# 输出：
# [[1,1,2],
#  [1,2,1],
#  [2,1,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [1,2,3]
# 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 8 
#  -10 <= nums[i] <= 10 
#  
#  Related Topics 数组 回溯 👍 913 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import Counter


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(c: Counter, s: List[int]):
            if sum(c.values()) == 0:
                ans.append(s[:])
                return
            for k, v in c.items():
                if v > 0:
                    s.append(k)
                    c[k] -= 1
                    dfs(c, s)
                    s.pop(-1)
                    c[k] += 1
        ans = []
        c = Counter(nums)
        print(c)
        dfs(c, [])
        return ans


if __name__ == '__main__':
    print(Solution().permuteUnique([1,3,2]))



# leetcode submit region end(Prohibit modification and deletion)
