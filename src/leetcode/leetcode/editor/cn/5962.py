from typing import List
from collections import Counter


class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        cnt = Counter()
        for word in words:
            cnt[word] += 1

        ans = 0
        set_middle = False
        for word, num in cnt.items():
            if word[0] != word[1]:
                word2 = word[1] + word[0]
                if word2 in cnt:
                    ans += 4*min(cnt[word], cnt[word2])
                    cnt[word] = 0
                    cnt[word2] = 0
            else:
                if cnt[word] % 2 == 0:
                    ans += 2*cnt[word]
                else:
                    if not set_middle:
                        ans += 2*cnt[word]
                        set_middle = True
                    else:
                        ans += 2*(cnt[word] - 1)
        return ans



if __name__ == '__main__':
    print(Solution().longestPalindrome(["cc","ll","xx"]))