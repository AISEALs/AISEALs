# 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。 
# 
#  算法的时间复杂度应该为 O(log (m+n)) 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums1 = [1,3], nums2 = [2]
# 输出：2.00000
# 解释：合并数组 = [1,2,3] ，中位数 2
#  
# 
#  示例 2： 
# 
#  
# 输入：nums1 = [1,2], nums2 = [3,4]
# 输出：2.50000
# 解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
#  
# 
#  示例 3： 
# 
#  
# 输入：nums1 = [0,0], nums2 = [0,0]
# 输出：0.00000
#  
# 
#  示例 4： 
# 
#  
# 输入：nums1 = [], nums2 = [1]
# 输出：1.00000
#  
# 
#  示例 5： 
# 
#  
# 输入：nums1 = [2], nums2 = []
# 输出：2.00000
#  
# 
#  
# 
#  提示： 
# 
#  
#  nums1.length == m 
#  nums2.length == n 
#  0 <= m <= 1000
#  0 <= n <= 1000 
#  1 <= m + n <= 2000 
#  -10⁶ <= nums1[i], nums2[i] <= 10⁶ 
#  
#  Related Topics 数组 二分查找 分治 👍 4846 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m = len(nums1)
        n = len(nums2)

        def getMedianOneList(nums: List[int]) -> float:
            if (m + n) % 2 == 1:
                return nums[(m + n) // 2]
            else:
                return (nums[(m + n) // 2 - 1] + nums[(m + n) // 2]) / 2

        if m == 0:
            nums = nums2 if m == 0 else nums1
            return getMedianOneList(nums)

        if nums2[-1] <= nums1[0]:
            nums1, nums2 = nums2, nums1
        if nums1[-1] <= nums2[0]:
            return getMedianOneList(nums1 + nums2)

        def getMedianTwoList(mid1: int, mid2: int) -> float:
            if (m + n) % 2 == 1:
                if mid1 < 0:
                    return nums2[mid2]
                elif mid2 < 0:
                    return nums1[mid1]
                else:
                    return max(nums1[mid1], nums2[mid2])
            else:
                if mid1 < 0:
                    max_left = nums2[mid2]
                elif mid2 < 0:
                    max_left = nums1[mid1]
                else:
                    max_left = max(nums1[mid1], nums2[mid2])
                if mid1 + 1 >= m:
                    min_right = nums2[mid2+1]
                elif mid2 + 1 >= n:
                    min_right = nums1[mid1+1]
                else:
                    min_right = min(nums1[mid1 + 1], nums2[mid2 + 1])
                return (max_left + min_right) / 2

        num_left = (m + n + 1) // 2
        left = -1
        right = m
        while left <= right:
            mid1 = (left + right) // 2
            mid2 = num_left - (mid1 + 1) - 1

            if mid1 >= 0 and mid1 + 1 < m and mid2 >= 0 and mid2 + 1 < n:
                if nums1[mid1] <= nums2[mid2 + 1] and nums2[mid2] <= nums1[mid1 + 1]:
                    return getMedianTwoList(mid1, mid2)
                elif nums1[mid1] > nums2[mid2] + 1:
                    right = mid1
                else:
                    left = mid1
            elif mid1 < 0:
                if nums2[mid2] <= nums1[mid1+1]:
                    return getMedianTwoList(mid1, mid2)
                else:
                    left = mid1
            elif mid1 + 1 >= m:
                if nums1[mid1] <= nums2[mid2+1]:    # 这里mid2+1必定不为空，否则nums2为[],这种情况最上面已经处理
                    return getMedianTwoList(mid1, mid2)
                else:
                    right = mid1
            elif mid2 < 0:
                if nums1[mid1] <= nums2[mid2+1]:
                    return getMedianTwoList(mid1, mid2)
                else:
                    right = mid1
            elif mid2 + 1 >= n:
                if nums2[mid2] <= nums1[mid1+1]:
                    return getMedianTwoList(mid1, mid2)
                else:
                    left = mid1
        return -1


if __name__ == '__main__':
    # nums1 = [1, 3]; nums2 = [2]
    # nums1 = [1, 2]; nums2 = [3, 4]
    # nums1 = [0, 0]; nums2 = [0, 0]
    # nums1 = []; nums2 = [1]
    # nums1 = [2]; nums2 = []
    # nums1 = [3]; nums2 = [1, 2, 4]
    nums1 = [1, 2, 4]; nums2 = [3]
    # nums1 = [4]; nums2 = [1, 2, 3, 5]
    # nums1 = [1, 2, 3, 5]; nums2 = [4]
    # nums1 = [1,3,4,5,6]; nums2= [2]
    print(Solution().findMedianSortedArrays(nums1=nums1, nums2=nums2))

# leetcode submit region end(Prohibit modification and deletion)
