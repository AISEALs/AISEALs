# 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重
# 复的三元组。 
# 
#  注意：答案中不可以包含重复的三元组。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-1,0,1,2,-1,-4]
# 输出：[[-1,-1,2],[-1,0,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [0]
# 输出：[]
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= nums.length <= 3000 
#  -10⁵ <= nums[i] <= 10⁵ 
#  
#  Related Topics 数组 双指针 排序 👍 4207 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import Counter


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        first = None
        for i in range(len(nums)):
            if first is not None and first == nums[i]:
                continue
            first = nums[i]
            second = None
            target = - nums[i]
            for j in range(i+1, len(nums)):
                if second is not None and second == nums[j]:
                    continue
                second = nums[j]
                l = j
                r = len(nums)
                while l + 1 != r:
                    mid = (l + r)//2
                    if nums[mid] == target - nums[j]:
                        ans.append([nums[i], nums[j], nums[mid]])
                        break
                    elif nums[mid] > target - nums[j]:
                        r = mid
                    else:
                        l = mid

        return ans

if __name__ == '__main__':
    print(Solution().threeSum(nums = [-1,0,1,2,-1,-4]))
# leetcode submit region end(Prohibit modification and deletion)
