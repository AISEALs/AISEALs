class Solution:
    def capitalizeTitle2(self, title: str) -> str:
        def upper(s: str):
            ret = s
            if 'a' < s and s < 'z':
                ret = chr(ord(s) - ord('a') + ord('A'))
            return ret
        def lower(s: str):
            ret = s
            if 'A' < s and s < 'Z':
                ret = chr(ord(s) - ord('A') + ord('a'))
            return ret
        title = [i for i in title]
        i = 0
        j = 0
        n = len(title)
        while i < n and j < n:
            while i < n and title[i] == ' ':
                i += 1
            j = i
            while j < n and title[j] != ' ':
                j += 1

            if j - i <= 2:
                for k in range(i, j):
                    title[k] = lower(title[k])
            else:
                title[i] = upper(title[i])
                for k in range(i+1, j):
                    title[k] = lower(title[k])
            i = j

        return ''.join(title)

    def capitalizeTitle(self, title: str) -> str:
        word_num = 0
        ans = []
        for i in range(len(title)):
            if title[i] == ' ':
                word_num = 0
            else:
                word_num += 1
                if word_num == 3:
                    ans[i - 2] = ans[i - 2].upper()
            ans.append(title[i].lower())
        return ''.join(ans)

if __name__ == '__main__':
    print(Solution().capitalizeTitle("First leTTeR of EACH Word"))