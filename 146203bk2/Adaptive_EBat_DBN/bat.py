#!/usr/bin/env python2
import random
from Adaptive_EBat_DBN.BatAlgorithm import *

def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

# For reproducive results
#random.seed(5)
def algm():
    Algorithm = BatAlgorithm(10, 40, 1000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, Fun)
    best = Algorithm.move_bat()
    return abs(best)
