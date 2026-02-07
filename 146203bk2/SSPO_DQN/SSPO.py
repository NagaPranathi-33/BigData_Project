import random,numpy as np

def algm():
    def Functions(x):
        fit = []
        for i in range(len(x)):
            a=0
            for j in range(len(x[i])):
                a += x[i][j]**4
            fit.append(a)
        return fit

    def initialize(m,n):
        data = []
        for i in range(m):
            tem = []
            for j in range(n):
                tem.append(random.randint(1,10))
            data.append(tem)
        return data

    student = 10
    var = 5
    Max_iteration=10
    x = initialize(student,var)
    fit = Functions(x)
    Best_fitness = min(fit)
    Best_ind = np.argmin(fit)
    Best_student = x[Best_ind]
    a0, b0, bmax = 1, 1, 5  # alpha & beta value
    t=0
    # loop begins
    overall_fit, overall_best = [], []
    while t<Max_iteration:
        itr = t + 1  # current iteration
        alpha = a0 - ((a0 / Max_iteration) * (itr))  # alpha update
        beta = b0 + (((bmax - b0) / Max_iteration) * (itr))  # beta update
        for i in range(var):
            sum = [0]*var
            mean = [0]*var
            for gw in range(var):
                for fi in range(student):
                    sum[gw] = sum[gw] + x[fi][gw]
                mean[gw] = sum[gw] / student

            check,mid = [],[]
            for a in range(student):
                check.append(random.random())
                mid.append(random.random())

            for ii in range(var):
                Xmean = np.mean(x[ii])

                if Best_fitness==min(fit):
                    k = random.randint(1,2)
                    Best_student[ii] = Best_student[ii]+((-1)**k)*random.random()*(Best_student[ii]-x[ii][i]) # Eq.1
                elif check[ii]<mid[ii]:
                    rta = random.random()
                    if rta > random.random():
                        x[ii][i] = Best_student[ii]+(random.random()*(Best_student[ii]-x[ii][i])) #  Eq.2a
                    else:
                        ########### Proposed updated equation SSPO ############
                        for j in range(len(x[ii])):
                            rand = random.random()
                            r = random.random()
                            x[ii][i] = (beta*rand/(beta*rand-r))*(((x[ii][i]+r*(alpha*rand*x[ii][j]-x[ii][i]*(1-rand*(beta+alpha))))/(beta*rand))-Xmean)
                an = random.random()
                if random.random()>an:
                    x[ii][i] = x[ii][i]+((random.random()*(Xmean-x[ii][i]))) # Eq.3
                else:
                    x[ii][i] = min(x[ii])+(random.random()*(max(x[ii])-min(x[ii]))) # eq.4
        fit = Functions(x)  # fitness
        bst = np.argmin(fit)
        overall_fit.append(fit[bst])
        overall_best.append(x[bst])
        Best_ind = np.argmin(fit)
        Best_student = x[Best_ind]
        t += 1
    Best_ind = np.argmin(fit)
    Best_student = x[Best_ind]
    return np.max(Best_student)














