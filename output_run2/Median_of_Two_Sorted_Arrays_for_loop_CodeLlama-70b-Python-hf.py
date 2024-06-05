from typing import List
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums = nums1 + nums2
        nums.sort()
        if len(nums) % 2 == 0:
            return (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2
        else:
            return nums[len(nums) // 2]

nums1 = [i for i in range(-10 ** 6, -10 ** 6 + 1000)]
nums2 = [i for i in range(10 ** 6 - 1000, 10 ** 6)]

solution_object = Solution()
assert solution_object.findMedianSortedArrays(nums1, nums2) == -0.50000