### MOEA/D-DE: Multi-objective evolutionary algorithm based on decomposition and differential evolution

##### Reference: Li H, Zhang Q. Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2008, 13(2): 284-302.

##### MOEA/D-DE is a classic multi-objective evolution algorithm (MOEA).

| Variables | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| npop      | Population size                                              |
| iter      | Iteration number                                             |
| lb        | Lower bound                                                  |
| ub        | Upper bound                                                  |
| T         | Neighborhood size (default = 20)                             |
| delta     | The probability that the parents are selected from neighborhood (default = 0.9) |
| nr        | The maximum solutions replaced by child (default = 2)        |
| CR        | Crossover rate (default = 1)                                 |
| F         | Mutation scalar number (default=0.5)                         |
| eta_m     | Spread factor distribution index (default = 20)              |
| nvar      | The dimension of decision space                              |
| nobj      | The dimension of objective space                             |
| V         | Weight vectors                                               |
| B         | The T closet weight vectors                                  |
| pop       | Population                                                   |
| objs      | Objectives                                                   |
| z         | Ideal point                                                  |
| off       | Offspring                                                    |
| off_obj   | Offspring objective                                          |
| c         | Update counter                                               |
| pf        | Pareto front                                                 |

#### Test problem: ZDT3



$$
\left\{
\begin{aligned}
&f_1(x)=x_1\\
&f_2(x)=g(x)\left[1-\sqrt{x_1/g(x)}-\frac{x_1}{g(x)}\sin(10\pi x_1)\right]\\
&f_3(x)=1+9\left(\sum_{i=2}^nx_i\right)/(n-1)\\
&x_i\in[0, 1], \qquad i=1,\cdots,n
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    main(200, 500, np.array([0] * 30), np.array([1] * 30))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/MOEA:D-DE/Pareto front.png)

```python
Iteration 20 completed.
Iteration 40 completed.
Iteration 60 completed.
Iteration 80 completed.
Iteration 100 completed.
Iteration 120 completed.
Iteration 140 completed.
Iteration 160 completed.
Iteration 180 completed.
Iteration 200 completed.
Iteration 220 completed.
Iteration 240 completed.
Iteration 260 completed.
Iteration 280 completed.
Iteration 300 completed.
Iteration 320 completed.
Iteration 340 completed.
Iteration 360 completed.
Iteration 380 completed.
Iteration 400 completed.
Iteration 420 completed.
Iteration 440 completed.
Iteration 460 completed.
Iteration 480 completed.
Iteration 500 completed.
```

