from typing import List


class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(n + 1):
            print('before', i, ans[i], i>>1, ans[i>>1], i & 1)
            ans[i] = ans[i>>1] + (i & 1)
            print('after', i, ans[i], i>>1, ans[i>>1], i & 1)
        return ans


print(Solution().countBits(3))
