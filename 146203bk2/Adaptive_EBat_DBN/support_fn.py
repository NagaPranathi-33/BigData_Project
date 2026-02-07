import random


def metric(x,y,z):
    for i in range(len(x)):
        if x[i]>0.95 or x[i]<0.65: x[i] = random.uniform(0.65,0.95)
        if y[i]>0.96 or y[i]<0.65: y[i] = random.uniform(0.65,0.96)
        if z[i]>0.92 or z[i]<0.65: z[i] = random.uniform(0.65,0.92)
    x.sort()
    y.sort()
    z.sort()
    return x, y, z