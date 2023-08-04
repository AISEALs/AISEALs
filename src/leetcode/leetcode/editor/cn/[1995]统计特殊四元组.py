# 给你一个 下标从 0 开始 的整数数组 nums ，返回满足下述条件的 不同 四元组 (a, b, c, d) 的 数目 ： 
# 
#  
#  nums[a] + nums[b] + nums[c] == nums[d] ，且 
#  a < b < c < d 
#  
# 
#  
# 
#  示例 1： 
# 
#  输入：nums = [1,2,3,6]
# 输出：1
# 解释：满足要求的唯一一个四元组是 (0, 1, 2, 3) 因为 1 + 2 + 3 == 6 。
#  
# 
#  示例 2： 
# 
#  输入：nums = [3,3,6,4,5]
# 输出：0
# 解释：[3,3,6,4,5] 中不存在满足要求的四元组。
#  
# 
#  示例 3： 
# 
#  输入：nums = [1,1,1,3,5]
# 输出：4
# 解释：满足要求的 4 个四元组如下：
# - (0, 1, 2, 3): 1 + 1 + 1 == 3
# - (0, 1, 3, 4): 1 + 1 + 3 == 5
# - (0, 2, 3, 4): 1 + 1 + 3 == 5
# - (1, 2, 3, 4): 1 + 1 + 3 == 5
#  
# 
#  
# 
#  提示： 
# 
#  
#  4 <= nums.length <= 50 
#  1 <= nums[i] <= 100 
#  
#  Related Topics 数组 枚举 👍 58 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import Counter


class Solution:
    def findTwoNum(self, nums: List[int], s) -> int:
        rs = 0
        for i in range(len(nums)):
            m = s - nums[i]
            for j in range(i+1, len(nums)):
                if m == nums[j]:
                    rs += 1
        return rs

    # def countQuadruplets(self, nums: List[int]) -> int:
    #     n = len(nums)
    #     rs = 0
    #     for i in range(n-1, 2, -1):     # range(max, min, -1)
    #         s = nums[i]
    #         for j in range(i-1, 1, -1):
    #             s1 = nums[j]
    #             rs += self.findTwoNum(nums[0: j], s-s1)
    #     return rs

    def countQuadruplets(self, nums: List[int]) -> int:
        rs = 0
        n = len(nums)
        for a in range(n):
            for b in range(a+1, n):
                cnt = Counter()
                for c in range(n-2, b, -1):
                    cnt[nums[c+1]] += 1
                    total = nums[a] + nums[b] + nums[c]
                    rs += cnt[total]
        return rs

if __name__ == '__main__':
    print(Solution().countQuadruplets([1,1,1,3,5]))
#     时间复杂度是O(n^4), 其中n是nums的长度，空间复杂度是O(1)
# 如何优化？
# 1. Counter哈希表 + 倒序生成哈希表
# 细节：
# 1. range 的左闭右开形式 + 数组的下表从0开始
# leetcode submit region end(Prohibit modification and deletion)
