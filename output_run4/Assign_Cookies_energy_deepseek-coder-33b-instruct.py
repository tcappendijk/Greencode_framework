
from typing import List
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:

        g.sort()
        s.sort()

        child_i = cookie_j = 0

        while child_i < len(g) and cookie_j < len(s):
            if g[child_i] <= s[cookie_j]:
                child_i += 1
            cookie_j += 1

        return child_i

g = [i for i in range(3 * (10 ** 4), 0, -1)]
s = [i for i in range(3 * (10 ** 4), 0, -1)]

solution_object = Solution()
anwser = solution_object.findContentChildren(g, s)

assert anwser == 30000




















































































































































































































































































































































































































































































































