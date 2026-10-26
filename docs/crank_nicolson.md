# Crank--Nicolson propagation

`CrankNicolson` implements a matrix-free implicit midpoint step for a linear
equation. If `linear_rhs(state, time)` represents `L(t) state`, one step solves

$$
\left(I-\frac{\Delta t}{2}L(t_{n+1/2})\right)\Psi_{n+1}
=
\left(I+\frac{\Delta t}{2}L(t_{n+1/2})\right)\Psi_n.
$$

For `linear_rhs = -1j * H`, this is the usual Crank--Nicolson TDSE step.

## Matrix-free solve

The left-hand side is exposed to SciPy as a `LinearOperator`; the complete
matrix is never assembled by `integrator-lib`. Select GMRES or BiCGSTAB:

```python
from integrator_lib import CrankNicolson

integrator = CrankNicolson(
    solver="bicgstab",
    rtol=0.0,
    atol=1.0e-10,
    maxiter=None,
    preconditioner=preconditioner,
)

new_state = integrator(state, time, time_step, linear_rhs)
```

A callable preconditioner receives an array with the original state shape and
returns an array of that shape. A SciPy `LinearOperator` with the flattened
state dimension is also accepted.

The most recent solve reports:

```python
integrator.last_iterations
integrator.last_residual_norm
```

Failure of the selected iterative solver raises `LinearSolveError`.

## Hermitian propagation and absorbers

For a time-independent Hermitian Hamiltonian and an accurately converged
linear solve, the Crank--Nicolson Cayley transform preserves the norm. A CAP
makes the Hamiltonian non-Hermitian; Crank--Nicolson remains applicable, but
the norm is then expected to decrease as probability is absorbed.

The Krylov tolerance controls the accuracy of the implicit linear solve. It
should not be interpreted as a direct bound on norm drift or physical
time-discretization error.

## Imaginary time

For imaginary-time propagation,

$$
\frac{d}{d\tau}\Psi(\tau)=-H\Psi(\tau),
$$

supply a real-valued derivative returning `-H @ state`. Real input then stays
in real arithmetic when the Hamiltonian action and preconditioner are real.
Normalization and ground-state convergence tests belong to the calling
ground-state solver.
