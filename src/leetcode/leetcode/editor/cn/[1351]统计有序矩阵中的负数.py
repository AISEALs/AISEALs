# 给你一个 m * n 的矩阵 grid，矩阵中的元素无论是按行还是按列，都以非递增顺序排列。 
# 
#  请你统计并返回 grid 中 负数 的数目。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
# 输出：8
# 解释：矩阵中共有 8 个负数。
#  
# 
#  示例 2： 
# 
#  
# 输入：grid = [[3,2],[1,0]]
# 输出：0
#  
# 
#  示例 3： 
# 
#  
# 输入：grid = [[1,-1],[-1,-1]]
# 输出：3
#  
# 
#  示例 4： 
# 
#  
# 输入：grid = [[-1]]
# 输出：1
#  
# 
#  
# 
#  提示： 
# 
#  
#  m == grid.length 
#  n == grid[i].length 
#  1 <= m, n <= 100 
#  -100 <= grid[i][j] <= 100 
#  
# 
#  
# 
#  进阶：你可以设计一个时间复杂度为 O(n + m) 的解决方案吗？ 
# 
#  
#  Related Topics 数组 二分查找 矩阵 👍 82 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
class Solution:
    # def countNegatives(self, grid: List[List[int]]) -> int:
    #     m = len(grid)
    #     n = len(grid[0])
    #     ans = 0
    #     for i in range(m):
    #         for j in range(n):
    #             if grid[i][j] < 0:
    #                 ans += n - j
    #                 break
    #     return ans

    def countNegatives(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        i = 0
        ans = 0
        for j in reversed(range(m)):
            while i < n:
                if grid[j][i] < 0:
                    ans += n - i
                    break
                i += 1
        return ans


if __name__ == '__main__':
    grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
    print(Solution().countNegatives(grid))
# 要点：
# 1. 把矩阵grid 画出图来，行、列分别用什么代表，想怎样循环遍历，所以对应的i和j都是什么意思
# 2. 如何利用 行列非递增这个性质，倒序查
# leetcode submit region end(Prohibit modification and deletion)
