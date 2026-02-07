import random
import math
import numpy as np

def algm():
    lb, ub = 1, 20  # lower and upper bounds
    N, M = 20, 10   # population size N, solution length M
    Max_iter = 2

    def bound(value):
        value = int(value)
        if value < lb or value > ub:
            value = random.randint(lb, ub)
        return value

    def generate_soln(n, m, Xmin, Xmax):
        return [[random.randint(Xmin, Xmax) for _ in range(m)] for _ in range(n)]

    def fitness(soln):
        fit = []
        for ind in soln:
            hr = random.random()
            score = sum([x * hr for x in ind])
            fit.append(score)
        return fit

    Position = generate_soln(N, M, lb, ub)
    Fit = fitness(Position)
    best_idx = np.argmin(Fit)
    Xbest = Position[best_idx]
    
    t = 0
    overall_fit, overall_best = [], []

    while t < Max_iter:
        for i in range(len(Position)):
            for j in range(len(Position[i])):
                a = 2 - t * (2 / Max_iter)
                a2 = -1 + t * (-1 / Max_iter)
                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * random.random() + 1
                p = random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * Xbest[j] - Position[i][j])
                        Position[i][j] = Xbest[j] - A * D
                    else:
                        rand_leader_index = math.floor(N * random.random())
                        X_rand = Position[rand_leader_index]
                        D_X_rand = abs(C * X_rand[j] - Position[i][j])
                        Position[i][j] = X_rand[j] - A * D_X_rand
                else:
                    D_ = abs(Xbest[j] - Position[i][j])
                    Position[i][j] = D_ * math.exp(b * l) * math.cos(l * 2 * math.pi) + Xbest[j]

                # Ensure bounds
                Position[i][j] = bound(Position[i][j])

        Fit = fitness(Position)
        best = np.argmin(Fit)
        overall_fit.append(Fit[best])
        overall_best.append(Position[best])
        Xbest = Position[best]
        t += 1

    best = np.argmin(overall_fit)
    BEST_SOLUTION = overall_best[best]

    # Normalize weights to range [-1, 1] for neural net compatibility
    norm_weights = np.interp(BEST_SOLUTION, (min(BEST_SOLUTION), max(BEST_SOLUTION)), (-1, 1))
    return norm_weights
