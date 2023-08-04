# 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。 
# 
#  请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。 
# 
#  
# 
#  示例 1: 
# 
#  
# 输入: [3,2,1,5,6,4] 和 k = 2
# 输出: 5
#  
# 
#  示例 2: 
# 
#  
# 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
# 输出: 4 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= k <= nums.length <= 10⁴ 
#  -10⁴ <= nums[i] <= 10⁴ 
#  
#  Related Topics 数组 分治 快速选择 排序 堆（优先队列） 👍 1446 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 目的 找到nums[0]应该的位置，使nums[left] < nums[0] <= nums[right]
        # 涉及到交换，所以用双指针
        def quickSort(l, r) -> int:
            s = l
            i = l + 1
            j = r
            while True:
                while i < r and nums[i] <= nums[s]:
                    i += 1
                while j > l and nums[s] <= nums[j]:
                    j -= 1  # j必须找到第一个能够<key的位置, 如果找不到，那么就直接定位到l
                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]
                else:
                    break
            nums[s], nums[j] = nums[j], nums[s]
            return j

        l = 0
        r = len(nums) - 1
        target = len(nums) - k
        while l < r:
            mid = quickSort(l, r)
            print(nums, '[', l, r, ']', mid, target)
            if mid == target:
                return nums[mid]
            elif mid < target:
                l = mid + 1
            else:
                r = mid - 1
        return nums[l]


if __name__ == '__main__':
    print(Solution().findKthLargest([7,6,5,4,3,2,1], 5))
# leetcode submit region end(Prohibit modification and deletion)
