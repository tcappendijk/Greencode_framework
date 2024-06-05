from typing import List
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:

        g.sort()
        s.sort()

        i = 0
        j = 0
        count = 0
        while i < len(g) and j < len(s):
            if g[i] <= s[j]:
                count += 1
                i += 1
                j += 1
            else:
                j += 1
        return count

g = [i for i in range(3 * (10 ** 4), 0, -1)]
s = [i for i in range(3 * (10 ** 4), 0, -1)]

solution_object = Solution()
anwser = solution_object.findContentChildren(g, s)

assert anwser == 30000

































































































































































































































































































































































































































































































