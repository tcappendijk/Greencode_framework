"""
This is a simple target function that takes an integer as input and returns the
sum of all numbers from 1 to the input integer.
"""

import sys

def targetFunction(iterations):
    itemCounter = 1
    result = 1
    for _ in range(iterations):
        result = result + itemCounter
    return result, itemCounter

res, i = targetFunction((int(sys.argv[1])))
print(res, i)
