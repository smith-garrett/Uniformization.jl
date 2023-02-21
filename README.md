# Uniformization

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://smith-garrett.github.io/Uniformization.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://smith-garrett.github.io/Uniformization.jl/dev/)
[![Build Status](https://github.com/smith-garrett/Uniformization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/smith-garrett/Uniformization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/smith-garrett/Uniformization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/smith-garrett/Uniformization.jl)
[![DOI](https://zenodo.org/badge/555271933.svg)](https://zenodo.org/badge/latestdoi/555271933)

Uniformization.jl is a package for solving continuous-time Markov processes using
uniformization, also known as randomization or Jensen's method. Solutions are approximated
by converting the continuous-time problem to a discrete-time problem paired with a counting
mechanism. Currently, standard uniformization, Erlangization or external uniformization, and
method based on discrete observation times (Yoon & Shanthikumar, 1989, *Probability in the
Engineering and Informational Sciences*) are implemented.

## Installing

## Usage

Begin by creating a generator matrix $\mathbf{Q}$ for the problem. The $i,j$-th entry
specifies the transition rate per unit time from state $j$ to state $i$. We also need to
specify the initial conditions, $\mathbf{p}(0)$, which must be a probabilty distribution
over all states. Using the `TransitionRateMatrix` type is optional; ensuring that other
types of matrices have the correct properties can also be done using the
`transitionratematrix` and `transitionratematrix!` functions.

```julia
using Uniformization.jl
Q = TransitionRateMatrix([-1.0 1 0; 1 -2 1; 0 1 -1])
p0 = [1.0, 0, 0]
```

From here, we can solve for the probability distribution at time $\mathbf{p}(t)$,
$\mathbf{p}(t) = e^{\mathbf{Q}t} \mathbf{p}(0)$:

```julia
t = 0.5
k = 2^8
uniformize(Q, p0, k, t)
```

The parameter $k$ controls the accuracy of the approximation; the default is $2^{10}$.

The `erlangization` method is the default, as it seems to be a good compromise between
efficiency and robustness on stiff problems.

Automatic differentiation using `ForwardDiff.jl` and `ReverseDiff.jl` should work as
expected.

Further information is available in the docstrings.

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).
