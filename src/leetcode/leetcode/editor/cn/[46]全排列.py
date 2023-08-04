# 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,2,3]
# 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [0,1]
# 输出：[[0,1],[1,0]]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [1]
# 输出：[[1]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 6 
#  -10 <= nums[i] <= 10 
#  nums 中的所有整数 互不相同 
#  
#  Related Topics 数组 回溯 👍 1729 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans = []

        def backtrace(left_nums: List[int], s: List[int]):
            if len(left_nums) == 0:
                ans.append(s[:])
                return
            for i in range(len(left_nums)):
                s.append(left_nums[i])
                backtrace(left_nums[:i] + left_nums[i+1:], s)
                s.pop(-1)

        backtrace(nums, [])
        return ans


if __name__ == '__main__':
    print(Solution().permute(list(range(1))))

# leetcode submit region end(Prohibit modification and deletion)
