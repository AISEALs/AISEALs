# 给你一个仅由整数组成的有序数组，其中每个元素都会出现两次，唯有一个数只会出现一次。 
# 
#  请你找出并返回只出现一次的那个数。 
# 
#  你设计的解决方案必须满足 O(log n) 时间复杂度和 O(1) 空间复杂度。 
# 
#  
# 
#  示例 1: 
# 
#  
# 输入: nums = [1,1,2,3,3,4,4,8,8]
# 输出: 2
#  
# 
#  示例 2: 
# 
#  
# 输入: nums =  [3,3,7,7,10,11,11]
# 输出: 10
#  
# 
#  
# 
#  
# 
#  提示: 
# 
#  
#  1 <= nums.length <= 10⁵ 
#  0 <= nums[i] <= 10⁵ 
#  
#  Related Topics 数组 二分查找 👍 302 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List

class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l = 0
        r = len(nums) - 1
        while l < r:
            mid = l + (r - l)//2
            if mid % 2 == 1:
                mid -= 1
            if nums[mid] == nums[mid+1]:
                l = mid + 2
            else:
                r = mid + 1
        return nums[l]


    def singleNonDuplicate2(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        l = 0
        r = len(nums) - 1
        while l < r:
            mid = l + (r - l)//2
            if nums[mid] == nums[mid+1]:
                if (r - mid) % 2 == 0:
                    l = mid + 2
                else:
                    r = mid - 1
            elif nums[mid] == nums[mid-1]:
                if (mid - l) % 2 == 0:
                    r = mid - 2
                else:
                    l = mid + 1
            else:
                return nums[mid]
        return nums[l]

if __name__ == '__main__':
    print(Solution().singleNonDuplicate(nums = [1, 1, 2, 2, 3]))
# leetcode submit region end(Prohibit modification and deletion)
