
for _ in range(100000):
    nums1 = [i for i in range(-10 ** 6, -10 ** 6 + 1000)]
    nums2 = [i for i in range(10 ** 6 - 1000, 10 ** 6)]

    solution_object = Solution()
    assert solution_object.findMedianSortedArrays(nums1, nums2) == -0.50000