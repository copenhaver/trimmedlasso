### Background

Sparse modeling in linear regression has been a topic of fervent interest in recent years. This interest has taken several forms, from substantial developments in the theory of the [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) to advances in algorithms for convex optimization. Throughout there has been a strong emphasis on the increasingly high-dimensional nature of linear regression problems; in such problems, where the number of variables `p` can vastly exceed the number of observations `n`, sparse modeling techniques are critical for performing inference.

One of the fundamental approaches to sparse modeling in the usual linear regression model of `y = Xβ + ε` is constrained best subset selection:
```
  minimize_β    0.5*norm(y-Xβ)^2
  subject to    nnz(β) <= k,
```
where `nnz` denotes the number of nonzero entries.

One can also consider the Lagrangian or penalized form, namely,
```
  minimize_β    0.5*norm(y-Xβ)^2 + λ*nnz(β)
```
for parameter λ > 0. One of the advantages of the former approach over the latter is that the former offers direct control over estimators’ sparsity via the discrete parameter `k`, as opposed to the Lagrangian form for which the correspondence between the continuous parameter `μ` and the resulting sparsity of estimators obtained is not entirely clear.



### Our approach

This page contains several sample implementations of an estimation procedure for sparse modeling as described in [Bertsimas, Copenhaver, and Mazumder, "The Trimmed Lasso: Sparsity and Robustness", arXiv preprint ("BCM17")](http://www.optimization-online.org/DB_HTML/2017/08/6167.html).

The problem that we focus on is as follows:
```
minimize_β    0.5*norm(y-X*β)^2 + μ*sum_i |β_i| + λ*T_k(β),
```
where `β` is the `p`-dimensional optimization variable, `μ,λ>0` are parameters, and `T_k` is the *trimmed Lasso*, namely,
```
T_k(β) = sum_{i=k+1}^p |β_{(i)}|
```
where `|β_{(1)}| >= |β_{(2)}| >= ... >= |β_{(p)}|` are the sorted entries of `β`. In other words, the trimmed Lasso penalty is the sum of the magnitudes of the smallest `p-k` entries of `β`.

For details on how the trimmed Lasso arises and why it is useful as a variable selection method, see BCM17 and references therein. One reason why it is useful, as shown previously, is that it is an *exact penalty method* for best subset selection; namely, for `λ` sufficiently large, the trimmed Lasso problem *exactly* solves the best subset selection problem
```
  minimize_β    0.5*norm(y-Xβ)^2 + μ*sum_i |β_i|
  subject to    nnz(β) <= k,
```

### Algorithms

The trimmed Lasso minimization problem above is NP-hard in general. Therefore, we include implementations of both exact algorithms (based on techniques in [mixed integer optimization](https://en.wikipedia.org/wiki/Integer_programming)) as well as fast heuristic algorithms based on popular techniques in convex optimization.

In particular, the different techniques that we detail are as follows:

1. Exact:
  * Formulation based on [SOS-1 constraints](https://en.wikipedia.org/wiki/Special_ordered_set)
  * Formulation using *big M* values

2. Heuristics:
  * [Alternating minimization](http://curtis.ml.cmu.edu/w/courses/index.php/Alternating_Minimization) (equivalently, sequential linearization)
  * [ADMM](http://stanford.edu/~boyd/admm.html)
  * [Convex envelopes](https://en.wikipedia.org/wiki/Lower_convex_envelope)

### Implementations

We provide implementations of the above algorithms in several languages:

1. [julia](./julia/)

  This implementation is the most complete and contains all of the algorithms above. The choice of `julia` is ideal for our purposes because of its flexibility in terms of optimization solvers using [JuMP](https://github.com/JuliaOpt/JuMP.jl). This implementation is also the most flexible: you can use open-source or commercial solvers for the integer programming problems, and you can specify your desired solver for usual Lasso problems (we include a basic implementation).

2. [MATLAB](./matlab/)

     This implementation is the most high-level. As such, it relies heavily on [`cvx`](https://cvxr.com/cvx/ "CVX") and [`Gurobi`](http://www.gurobi.com).

3. [R](./R/)



### Citation

If you would like to cite this work, please use the following citation for BCM17:
```
@article{bcm17,
  author  = {Dimitris Bertsimas and Martin S. Copenhaver and Rahul Mazumder},
  title   = {The Trimmed Lasso: Sparsity and Robustness},
  year    = {2017},
  url     = {http://arxiv.org/abs/***}
}
```
