import sys
from typing import List
import bisect

ans = []

def getMusicNum(music_lens: List[int], q_list: List[List[int]]):
    pre_sum = [music_lens[0]]
    for i in music_lens[1:]:
        pre_sum.append(pre_sum[-1] + i)
    time = 0
    for op, x in q_list:
        if op == 1:
            music_lens.append(x)
            pre_sum.append(pre_sum[-1] + x)
        if op == 2:
            time += x
            # left = time % (pre_sum[-1])     # 这样写不对，应该是time溢出了！！！
            time %= pre_sum[-1]
            x = bisect.bisect(pre_sum, time)
            # ans.append(x + 1)
            print(x + 1)


def read_data(source):
    N = int(next(source))
    for _ in range(N):
        num_rows = int(next(source))
        music_lens = next(source).split()
        music_lens = [int(s) for s in music_lens]
        q = int(next(source))
        q_list = [[int(x) for x in next(source).split()] for _ in range(q)]
        yield music_lens, q_list


for music_lens, q_list in read_data(sys.stdin):
    getMusicNum(music_lens, q_list)
