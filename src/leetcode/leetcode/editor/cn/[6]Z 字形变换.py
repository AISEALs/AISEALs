# 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。 
# 
#  比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下： 
# 
#  
# P   A   H   N
# A P L S I I G
# Y   I   R 
# 
#  之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。 
# 
#  请你实现这个将字符串进行指定行数变换的函数： 
# 
#  
# string convert(string s, int numRows); 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "PAYPALISHIRING", numRows = 3
# 输出："PAHNAPLSIIGYIR"
#  
# 示例 2：
# 
#  
# 输入：s = "PAYPALISHIRING", numRows = 4
# 输出："PINALSIGYAHRPI"
# 解释：
# P     I    N
# A   L S  I G
# Y A   H R
# P     I
#  
# 
#  示例 3： 
# 
#  
# 输入：s = "A", numRows = 1
# 输出："A"
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 1000 
#  s 由英文字母（小写和大写）、',' 和 '.' 组成 
#  1 <= numRows <= 1000 
#  
#  Related Topics 字符串 👍 1445 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
import numpy as np


class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        state = [(1, 0), (-1, 1)]
        flag = 0
        z = np.zeros([numRows, 1000], dtype=str)
        i, j = (0, 0)
        for c in s:
            z[i, j] = c
            i, j = i + state[flag][0], j + state[flag][1]
            if i == numRows - 1 or i == 0:
                flag ^= 1
        ans = []
        for line in z:
            for i in line:
                if i != '':
                    ans.append(i)
        return ''.join(ans)

    def convert2(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        ss = ['' for _ in range(numRows)]
        i, flag = 0, -1
        for i, c in enumerate(s):
            ss[i] += c
            if i % (numRows - 1) == 0:
                flag = -flag
            i += flag
        return ''.join(ss)


if __name__ == '__main__':
    print(Solution().convert(s = "AB", numRows=1))

# leetcode submit region end(Prohibit modification and deletion)
