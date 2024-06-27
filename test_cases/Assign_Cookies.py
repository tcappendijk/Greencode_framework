

for _ in range(600):
    g = [i for i in range(3 * (10 ** 4), 0, -1)]
    s = [i for i in range(3 * (10 ** 4), 0, -1)]
    solution_object = Solution()
    anwser = solution_object.findContentChildren(g, s)

    assert anwser == 30000