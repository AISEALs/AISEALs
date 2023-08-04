from collections import defaultdict
from typing import List


class Solution:
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        s = set()
        for word in startWords:
            mask = 0
            for ch in word:
                mask |= 1 << (ord(ch) - ord('a'))
            s.add(mask)
        ans = 0
        for word in targetWords:
            mask = 0
            for ch in word:
                mask |= 1 << (ord(ch) - ord('a'))
            for ch in word:
                if mask ^ (1 << (ord(ch) - ord('a'))) in s:  # 去掉这个字符
                    ans += 1
                    break
        return ans


if __name__ == '__main__':
    print(Solution().wordCount(startWords = ["ab","a"], targetWords = ["abc","abcd"]))
