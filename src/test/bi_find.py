from typing import List


class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def getMaxNumByDeleteKNum(nums: List[int], k) -> List[int]:
            # print(nums, k)
            stack = []
            remain = len(nums) - k
            for num in nums:
                while k and stack and stack[-1] < num:
                    stack.pop(-1)
                    k -= 1
                stack.append(num)
            # print(stack[:(remain)])
            return stack[:remain]

        def compareList(l1: List[int], i: int, l2: List[int], j: int) -> int:
            if i >= len(l1) and j >= len(l2):
                return 0
            if i >= len(l1):
                return -1
            if j >= len(l2):
                return 1

            while i < len(l1) or j < len(l2):
                if l1[i] != l2[j]:
                    return l1[i] - l2[j]
                i += 1
                j += 1
                if i >= len(l1):
                    return -1
                if j >= len(l2):
                    return 1
            return 0

        ans_list = []
        for k1 in range(k+1):
            if len(nums1) < k1 or len(nums2) < k - k1:
                continue
            l1 = getMaxNumByDeleteKNum(nums1, len(nums1) - k1)
            l2 = getMaxNumByDeleteKNum(nums2, len(nums2) - (k - k1))
            start1 = 0
            start2 = 0
            merge_list = []
            while True:
                rt = compareList(l1, start1, l2, start2)
                if rt == 0:
                    break
                elif rt > 0:
                    merge_list.append(l1[start1])
                    start1 += 1
                else:
                    merge_list.append(l2[start2])
                    start2 += 1
            print(len(l1), len(l2), len(merge_list), start1, start2, l1, l2, merge_list)
            if compareList(ans_list, 0, merge_list, 0) < 0:
                ans_list = merge_list
        return ans_list


if __name__ == '__main__':
    # nums1 = [6,7]
    # nums2 = [6,0,4]
    # k = 5
    # nums1 = [3, 4, 6, 5]
    # nums2 = [9, 1, 2, 5, 8, 3]
    # k = 5
    nums1 = [8,1,8,8,6]
    nums2 = [4]
    k = 2
    print(len(nums1), len(nums2), k)
    # [9, 8, 6, 5, 3]
    res = Solution().maxNumber(nums1, nums2, k)
    print(res)
    # part_res =[2,1,1,1,0,2,1,2,2,2,2,0,1,0,0,2,0,2,0,2,1,0,1,1,0,1,0,1,2,1,1,1,0,1,2,2,1,0,0,1,2,1,2,2,1,1,0,1,2,0,2,0,1,2,0,2,1,1,1,2,0,0,1,1,0,2,1,0,1,2,1,0,2,2,1,0,2,0,1,1,0,0,2,2,0,1,0,2,0,2,2,2,2,1,1,1,1,0,0,1,0,2,1,2,0,1,0,0,0,1,2,1,0,1,1,2,0,2,2,0,0,1,1,2,2,1,1,2,2,1,0,1,2,0,1,2,2,0,0,0,2,0,2,0,2,2,0,1,1,1,1,2,2,2,2,0,0,2,2,2,2,0,2,0,1,0,0,2,1,0,0,2,0,2,1,1,1,1,0,1,2,0,2,1,0,1,1,1,0,0,2,2,2,0,2,1,1,1,2,2,0,0,2,2,2,2,2,0,2,0,2,0,2,0,0,1,0,1,1,0,0,2,1,1,2,2,2,1,2,2,0,0,2,1,0,2,1,2,1,1,1,0,2,0,1,1,2,1,1,0,0,1,0,1,2,2,2,0,2,2,1,0,1,2,1,2,0,2,2,0,1,2,2,1,2,2,1,1,2,2,2,2,2,1,2,0,1,1,1,2,2,2,0,2,0,2,0,2,1,1,0,2,2,2,1,0,2,1,2,2,2,0,1,1,1,1,1,1,0,0,0,2,2,0,1,2,1,0,0,2,2,2,2,1,0,2,0,1,2,0,0,0,0,2,1,0,2,1,1,2,1,2,2,0,2,1,0,2,0,0,2,0,2,2,1,0,1,0,0,2,1,1,1,2,2,0,0,0,1,1,2,0,2,2,0,1,0,2,1,0,2,1,1,1,0,1,1,2,0,2,0,1,1,2,0,2,0,1,2,1,0,2,0,1,0,0,0,1,2,1,2,0,1,2,2,1,1,0,1,2,1,0,0,1,0,2,2,1,2,2,0,0,0,2,0,0,0,1,0,2,0,2,1,0,0,1,2,0,1,1,0,1,0,2,2,2,1,1,0,1,1,2,1,0,2,2,2,1,2,2,2,2,0,1,1,0,1,2,1,2,2,0,0,0,0,0,1,1,1,2]
    # print(res[0: len(part_res)] == part_res)