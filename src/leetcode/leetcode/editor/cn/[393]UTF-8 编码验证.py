# 给定一个表示数据的整数数组 data ，返回它是否为有效的 UTF-8 编码。 
# 
#  UTF-8 中的一个字符可能的长度为 1 到 4 字节，遵循以下的规则： 
# 
#  
#  对于 1 字节 的字符，字节的第一位设为 0 ，后面 7 位为这个符号的 unicode 码。 
#  对于 n 字节 的字符 (n > 1)，第一个字节的前 n 位都设为1，第 n+1 位设为 0 ，后面字节的前两位一律设为 10 。剩下的没有提及的二进制
# 位，全部为这个符号的 unicode 码。 
#  
# 
#  这是 UTF-8 编码的工作方式： 
# 
#  
#       Number of Bytes  |        UTF-8 octet sequence
#                        |              (binary)
#    --------------------+---------------------------------------------
#             1          | 0xxxxxxx
#             2          | 110xxxxx 10xxxxxx
#             3          | 1110xxxx 10xxxxxx 10xxxxxx
#             4          | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
#  
# 
#  x 表示二进制形式的一位，可以是 0 或 1。 
# 
#  注意：输入是整数数组。只有每个整数的 最低 8 个有效位 用来存储数据。这意味着每个整数只表示 1 字节的数据。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：data = [197,130,1]
# 输出：true
# 解释：数据表示字节序列:11000101 10000010 00000001。
# 这是有效的 utf-8 编码，为一个 2 字节字符，跟着一个 1 字节字符。
#  
# 
#  示例 2： 
# 
#  
# 输入：data = [235,140,4]
# 输出：false
# 解释：数据表示 8 位的序列: 11101011 10001100 00000100.
# 前 3 位都是 1 ，第 4 位为 0 表示它是一个 3 字节字符。
# 下一个字节是开头为 10 的延续字节，这是正确的。
# 但第二个延续字节不以 10 开头，所以是不符合规则的。
#  
# 
#  
# 
#  提示: 
# 
#  
#  1 <= data.length <= 2 * 10⁴ 
#  0 <= data[i] <= 255 
#  
#  Related Topics 位运算 数组 👍 170 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Automoton:
    def __init__(self):
        self.ans = True
        self.status = "start"
        self.byte_num = 0
        self.table = {
            "start": ["s1", "s21"],
            "s1": ["s1", "s21", "end"],
            "s21": ["s22", "end"],
            "s22": ["s1", "s21", "s22", "end"],
            "end": ["end"]
        }

    def equal(self, b, s):
        ret = True
        for c in s:
            c = int(c)
            c2 = b >> 7
            b = (b << 1) & ((1 << 8) - 1)
            if c != c2:
                ret = False
                break
        return ret

    def cal_byte_num(self, b):
        num = 0
        while True:
            c = b >> 7
            if c:
                num += 1
            else:
                break
            b = (b << 1) & ((1 << 8) - 1)
        return num - 1

    def get_byte(self, b):
        if self.status == 'start':
            if self.equal(b, "0"):
                return 0
            else:   # self.equal(b, "1")
                self.byte_num = self.cal_byte_num(b)
                return 1
        elif self.status == 's1':
            if self.equal(b, "0"):
                return 0
            else:   # if self.equal(b, "1"):
                self.byte_num = self.cal_byte_num(b)
                return 1
        elif self.status == 's21':
            if self.byte_num < 1 or self.byte_num > 3:
                return 1
            elif self.equal(b, '10'):
                self.byte_num -= 1
                return 0
            else:
                return 1
        elif self.status == 's22':
            if self.byte_num == 0:
                if self.equal(b, '0'):
                    return 0
                else:  # self.equal(b, '1')
                    self.byte_num = self.cal_byte_num(b)
                    return 1
            else:   # self.byte_num > 0
                if self.equal(b, '10'):
                    self.byte_num -= 1
                    return 2
                else:
                    return 3
        else:   # self.status == 'end'
            return 0

    def get(self, i):
        self.status = self.table[self.status][self.get_byte(i)]
        if self.status == 's1':
            self.ans = True
        elif self.status == 's21':
            self.ans = False
        elif self.status == 's22':
            self.ans = self.byte_num == 0
        elif self.status == 'end':
            self.ans = False
        # print(self.status, self.ans)


class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        automoton = Automoton()
        for i in data:
            print(bin(i))
            automoton.get(i)
        return automoton.ans


if __name__ == '__main__':
    print(Solution().validUtf8([250,145,145,145,145]))

# leetcode submit region end(Prohibit modification and deletion)
