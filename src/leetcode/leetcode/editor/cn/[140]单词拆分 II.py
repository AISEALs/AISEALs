# 给定一个字符串 s 和一个字符串字典 wordDict ，在字符串 s 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。以任意顺序 返回所有这些可
# 能的句子。 
# 
#  注意：词典中的同一个单词可能在分段中被重复使用多次。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
# 输出:["cats and dog","cat sand dog"]
#  
# 
#  示例 2： 
# 
#  
# 输入:s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine",
# "pineapple"]
# 输出:["pine apple pen apple","pineapple pen apple","pine applepen apple"]
# 解释: 注意你可以重复使用字典中的单词。
#  
# 
#  示例 3： 
# 
#  
# 输入:s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
# 输出:[]
#  
# 
#  
# 
#  提示： 
# 
#  
# 
#  
#  1 <= s.length <= 20 
#  1 <= wordDict.length <= 1000 
#  1 <= wordDict[i].length <= 10 
#  s 和 wordDict[i] 仅有小写英文字母组成 
#  wordDict 中所有字符串都 不同 
#  
#  Related Topics 字典树 记忆化搜索 哈希表 字符串 动态规划 回溯 👍 559 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from src.utils.timeit import timeit
from functools import lru_cache


class Solution:
    '''
    1: 动态规划：自底向上，
    2: 不带记忆的回溯
    2: 回溯：自顶向下，优势：不可拆分情况做剪枝
    '''
    @timeit
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # dp: d[i] = d[j] && check(word[j:i+1])
        d = [[] for _ in range(len(s))]
        for i in range(len(s)):
            for j in range(i+1):
                if (j == 0 or len(d[j-1]) > 0) and s[j:i+1] in wordDict:
                    if j == 0:
                        d[i].append(s[j:i+1])
                    else:
                        for word in d[j-1]:
                            d[i].append(' '.join([word, s[j:i+1]]))
        return d[-1]

    @timeit
    def wordBreak2(self, s: str, wordDict: List[str]) -> List[str]:
        def backtrace(word: str, path: List[str]):
            if not word:
                ans.append(' '.join(path))
                return
            # word[:len(word)]
            for i in range(1, len(word)+1):
                if word[:i] in wordDict:
                    path.append(word[:i])
                    backtrace(word[i:], path)
                    path.pop(-1)

        ans = []
        backtrace(s, [])
        return ans

    @timeit
    def wordBreak3(self, s: str, wordDict: List[str]) -> List[str]:
        @lru_cache(maxsize=None)
        def backtrace(word) -> List[List[str]]:
            if not word:
                return [[]]
            ans = []
            for i in range(1, len(word)+1):
                if word[:i] in wordDict:
                    left_ans = backtrace(word[i:])
                    for ls in left_ans:
                        ans.append([word[:i]] + ls)
            return ans
        ans = backtrace(s)
        print(backtrace.cache_info())
        return [' '.join(i) for i in ans]


if __name__ == '__main__':
    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    # s = "aaaaaaaaaaaaaaaaaaaaaaa"
    # wordDict = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa"]
    # s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    # wordDict = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"]
    res3 = Solution().wordBreak3(s, wordDict)
    res2 = Solution().wordBreak2(s, wordDict)
    res1 = Solution().wordBreak(s, wordDict)
# leetcode submit region end(Prohibit modification and deletion)
