import sys
from functools import cache


# 1. 第二行输入 n 个整数,其中 ai 表示第 i 个文件感染的病毒编号。
# 2. 第四行输入 (m+1) 个整数, c0表示一次万能杀毒需要的时间，ci 表示使用大小为 i 的病毒库的特效杀毒需要的时间。

def read_data(source):
    N = int(next(source))
    for _ in range(N):
        n = int(next(source))
        a_lst = [int(i) for i in next(source).split()]
        print('a_lst', a_lst)
        q = int(next(source))
        q_lst = [int(i) for i in next(source).split()]
        print('q_list:', q_lst)
        yield a_lst, q_lst


def solve(n_lst, q_lst):
    print(n_lst, q_lst)
    q_len = len(q_lst) - 1

    @cache
    def dfs(idx) -> int:
        if idx >= len(n_lst):
            print('-----------')
            return 0

        ans = float('inf')
        st = {n_lst[idx]}
        i = idx + 1
        while len(st) <= q_len:
            if i == len(n_lst) or n_lst[i] not in st:
                print(f"use 病毒库：{st}, 清除了{n_lst[idx: i]}, cost: {q_lst[len(st)]}")
                ans = min(ans, dfs(i) + q_lst[len(st)])
                if i == len(n_lst):
                    break
                st.add(n_lst[i])
            i += 1
        ans = min(ans, dfs(idx + 1) + q_lst[0])
        return ans

    print(dfs(0))


for x in read_data(sys.stdin):
    solve(x[0], x[1])
# solve([1, 2, 3, 4, 1, 1, 1], [4, 5, 6, 7])
