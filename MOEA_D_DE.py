#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/23 11:25
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MOEA_D_DE.py
# @Statement : Multi-objective evolutionary algorithm based on decomposition and differential evolution (MOEA/D-DE)
# @Reference : Li H, Zhang Q. Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2008, 13(2): 284-302.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform


def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from a n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, dim):
    # calculate approximately npop uniformly distributed reference points on dim dimensions
    h1 = 0
    while combination(h1 + dim, dim - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + dim), dim - 1))) - np.arange(dim - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < dim:
        h2 = 0
        while combination(h1 + dim - 1, dim - 1) + combination(h2 + dim, dim - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + dim), dim - 1))) - np.arange(dim - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * dim)
            points = np.concatenate((points, temp_points), axis=0)
    points = np.where(points != 0, points, 1e-3)
    return points


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def mutation(individual, lb, ub, dim, pm, eta_m):
    # polynomial mutation
    if np.random.random() < pm:
        site = np.random.random(dim) < 1 / dim
        mu = np.random.random(dim)
        delta1 = (individual - lb) / (ub - lb)
        delta2 = (ub - individual) / (ub - lb)
        temp = np.logical_and(site, mu <= 0.5)
        individual[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
        temp = np.logical_and(site, mu > 0.5)
        individual[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
        individual = np.where(((individual >= lb) & (individual <= ub)), individual, np.random.uniform(lb, ub))
    return individual


def main(npop, iter, lb, ub, T=20, delta=0.9, nr=2, CR=1, F=0.5, eta_m=20):
    """
    The main function
    :param npop: population number
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param T: neighborhood size (default = 20)
    :param delta: the probability that the parents are selected from neighborhood (default = 0.9)
    :param nr: the maximum solutions replaced by child (default = 2)
    :param CR: crossover rate (default = 1)
    :param F: mutation scalar number (default=0.5)
    :param eta_m: spread factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    nobj = len(cal_obj((lb + ub) / 2))  # the dimension of objective space
    V = reference_points(npop, nobj)  # weight vectors
    sigma = squareform(pdist(V, metric='euclidean'), force='no', checks=True)  # distances between weight vectors
    B = np.argsort(sigma)[:, : T]  # the T closet weight vectors
    npop = V.shape[0]  # population size
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = np.array([cal_obj(x) for x in pop])  # objectives
    z = np.min(objs, axis=0)  # ideal point

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        for i in range(npop):

            # Step 2.1. Select parents
            if np.random.random() < delta:
                P = np.random.permutation(B[i])
            else:
                P = np.random.permutation(npop)

            # Step 2.2. Differential evolution
            if np.random.random() < CR:
                off = pop[i] + F * (pop[P[0]] - pop[P[1]])
                off = np.where(((off >= lb) & (off <= ub)), off, np.random.uniform(lb, ub))
            else:
                off = pop[i].copy()

            # Step 2.3. Mutation
            off = mutation(off, lb, ub, nvar, 1, eta_m)
            off_obj = cal_obj(off)

            # Step 2.4. Update the ideal point
            z = np.min((z, off_obj), axis=0)

            # Step 2.5. Update neighbor solutions
            c = 0  # update counter
            for j in P:
                if c == nr:
                    break
                if np.max(V[j] * np.abs(off_obj - z)) < np.max(V[j] * np.abs(objs[j] - z)):
                    c += 1
                    pop[j] = off
                    objs[j] = off_obj

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    plt.figure()
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of ZDT3')
    plt.show()


if __name__ == '__main__':
    main(200, 500, np.array([0] * 30), np.array([1] * 30))
