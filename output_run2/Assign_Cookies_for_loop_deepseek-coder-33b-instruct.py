
from typing import List
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:

        g.sort()
        s.sort()

        child = cookie = 0
        while s and g:
            if s[-1] >= g[-1]:
                s.pop()
                g.pop()
                child += 1
            else:
                g.pop()
        return child

g = [i for i in range(3 * (10 ** 4), 0, -1)]
s = [i for i in range(3 * (10 ** 4), 0, -1)]

solution_object = Solution()
anwser = solution_object.findContentChildren(g, s)

assert anwser == 30000