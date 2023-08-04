from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # f[i][j] = f[i-1][j - cnt * c[i]]
        candidates.sort()
        # f = [0] * (target + 1)
        # f[0] = 1
        # for x in candidates:
        #     for j in range(target + 1):
        #         if j >= x:
        #             f[j] += f[j - x]
        # return f[target]
        n = len(candidates)

        def dfs(idx, target):
            if target < 0:
                return []
            if idx < 0:
                if target == 0:
                    return [[]]
                else:
                    return []

            res = []
            for i in range(idx, -1, -1):
                subs = dfs(i, target - candidates[idx])
                for sub in subs:
                    sub.append(candidates[idx])
                res.extend(subs)
            return res

        return dfs(n - 1, target)


if __name__ == '__main__':
    print(Solution().combinationSum([2, 3, 6, 7], 7))
    # print(Solution().combinationSum([2, 3, 5], 8))
