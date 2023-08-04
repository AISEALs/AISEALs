from typing import List
import numpy as np


class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        n = len(nums)
        one_cnt = sum(nums)
        ans = np.inf
        zero_cnt = None
        left = 0
        right = left + one_cnt
        # 开一个one_cnt windows, [l, right)
        while left <= n:
            if zero_cnt is None:
                zero_cnt = one_cnt - sum(nums[left:right])
            else:
                if nums[left % n] == 0:
                    zero_cnt -= 1
                if nums[right % n] == 0:
                    zero_cnt += 1
            ans = min(ans, zero_cnt)
            print('windows:', [nums[ii % n] for ii in range(left, right)], "zero_cnt:", zero_cnt, 'ans:', ans)
            left += 1
            right += 1
        return ans


if __name__ == '__main__':
    print(Solution().minSwaps([0, 1, 1, 1, 0, 0, 1, 1, 0]))
