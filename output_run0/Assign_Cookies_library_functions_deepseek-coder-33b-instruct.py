
from typing import List
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:

        g.sort()
        s.sort()

        child = cookie = 0
        while child < len(g) and cookie < len(s):
            if g[child] <= s[cookie]:
                child += 1
            cookie += 1

        return child

for _ in range(600):
    g = [i for i in range(3 * (10 ** 4), 0, -1)]
    s = [i for i in range(3 * (10 ** 4), 0, -1)]
    solution_object = Solution()
    anwser = solution_object.findContentChildren(g, s)

    assert anwser == 30000










































































































































































































































































































































































































































































































































