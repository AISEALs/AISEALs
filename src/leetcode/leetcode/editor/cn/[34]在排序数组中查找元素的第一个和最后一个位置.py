# 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。 
# 
#  如果数组中不存在目标值 target，返回 [-1, -1]。 
# 
#  进阶： 
# 
#  
#  你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？ 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [5,7,7,8,8,10], target = 8
# 输出：[3,4] 
# 
#  示例 2： 
# 
#  
# 输入：nums = [5,7,7,8,8,10], target = 6
# 输出：[-1,-1] 
# 
#  示例 3： 
# 
#  
# 输入：nums = [], target = 0
# 输出：[-1,-1] 
# 
#  
# 
#  提示： 
# 
#  
#  0 <= nums.length <= 10⁵ 
#  -10⁹ <= nums[i] <= 10⁹ 
#  nums 是一个非递减数组 
#  -10⁹ <= target <= 10⁹ 
#  
#  Related Topics 数组 二分查找 👍 1379 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0 or target < nums[0] or target > nums[-1]:
            return [-1, -1]
        left = right = -1
        l = -1
        r = len(nums)
        while l + 1 != r:
            mid = int((l + r) / 2)
            if nums[mid] < target:
                l = mid
            else:
                r = mid
        if nums[r] == target:
            left = r
        l = -1
        r = len(nums)
        while l + 1 != r:
            mid = int((l + r) / 2)
            if nums[mid] <= target:
                l = mid
            else:
                r = mid
        if nums[l] == target:
            right = l
        return [left, right]

if __name__ == '__main__':
    print(Solution().searchRange(nums = [5, 7 ,7 ,8, 8, 10], target = 6))

# leetcode submit region end(Prohibit modification and deletion)
