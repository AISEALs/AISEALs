from typing import List
import random


class Solution:
    def wiggleSort(self, nums: List[int], k: int) -> None:
        def partition(arr: List[int], left: int, right: int):
            idx = random.randint(left, right)
            arr[left], arr[idx] = arr[idx], arr[left]
            pivot = arr[left]
            while left < right:
                while left < right and arr[right] >= pivot:
                    right -= 1
                arr[left] = arr[right]
                while left < right and arr[left] <= pivot:
                    left += 1
                arr[right] = arr[left]
            arr[left] = pivot
            return left

        # def quickSort(arr: List[int], left: int, right: int):
        #     if left < right:
        #         idx = partition(arr, left, right)
        #         quickSort(arr, left, idx-1)
        #         quickSort(arr, idx+1, right)

        # quickSort(nums, 0, len(nums) - 1)
        def topKSplit(arr: List[int], left: int, right: int, k: int):
            mid = partition(arr, left, right)
            if mid == k:
                return arr[mid]
            elif mid < k:
                return topKSplit(arr, mid + 1, right, k - mid)
            else:
                return topKSplit(arr, left, mid - 1, k)

        print(topKSplit(nums, 0, len(nums) - 1, len(nums) - k))


nums = list(range(9))
random.shuffle(nums)
print(nums)
nums = [3,2,1,5,6,4]
k = 2

Solution().wiggleSort(nums, k)