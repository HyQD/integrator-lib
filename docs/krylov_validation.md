# SIL and SIA validation

The SIL and SIA implementations are validated on deterministic systems small
enough for direct matrix exponentiation. These checks separate Krylov
projection accuracy from the spatial discretization and physics of a
particular application.

Run the reference calculation from the repository root with:

```bash
python examples/validate_krylov_propagators.py
```

The same expectations are enforced by `tests/test_krylov_validation.py`.

## Constant Hermitian Hamiltonian

The first problem propagates a normalized complex state under a constant
four-dimensional Hermitian matrix. The reference is evaluated directly with
`scipy.linalg.expm`.

The four-dimensional SIL and SIA Krylov spaces span the complete problem, so
both exponential propagators reproduce the reference to floating-point
precision and preserve the norm. Crank--Nicolson preserves the norm while its
Cayley approximation converges to the exact exponential with second-order
global error.

For time steps `0.2`, `0.1`, and `0.05`, the Crank--Nicolson state errors are
approximately:

| Time step | State error |
| ---: | ---: |
| 0.20 | `1.89e-3` |
| 0.10 | `4.76e-4` |
| 0.05 | `1.19e-4` |

Halving the step reduces the error by approximately four, as expected for a
second-order method.

## Imaginary-time Hermitian exponential

The second Hermitian reference shifts the same matrix to a positive spectrum
and evaluates `exp(-tau * H) @ state` directly. SIL is configured with
`generator_factor=-1.0`; SIA consumes the generator `-H` without additional
configuration. Both reproduce the exact matrix exponential to floating-point
precision when the four-dimensional Krylov space spans the complete problem.

The state norm decreases in this test by construction. The validation does
not renormalize after each step because its purpose is to verify the
exponential action itself. Ground-state solvers may normalize and evaluate
their convergence criteria after accepting each imaginary-time step.

## CAP-like non-Hermitian generator

The CAP-like problem adds a nonnegative diagonal absorber $W$ and propagates
with

$$
L=-iH-W.
$$

The exact final norm is approximately `0.7974133840`, rather than one. SIA
reproduces both the state and this norm loss to floating-point precision when
the complete four-dimensional Arnoldi space is available.

Crank--Nicolson remains applicable and converges quadratically to the same
non-Hermitian exponential. SIL is intentionally excluded because the
effective Hamiltonian is not Hermitian.

## Time-dependent midpoint sampling

The final problem uses the commuting time-dependent Hamiltonian

$$
H(t)=\left(1+0.3t+0.2t^2\right)H_0.
$$

Because every Hamiltonian is proportional to the same constant matrix, the
exact result follows from the analytic integral of the scalar prefactor. This
makes it possible to test temporal midpoint sampling independently of a
reference ODE solver.

SIL, SIA, and Crank--Nicolson all show second-order convergence. Representative
SIL errors for 4, 8, 16, and 32 time steps are:

| Steps | State error | Error ratio after refinement |
| ---: | ---: | ---: |
| 4 | `6.53e-4` | -- |
| 8 | `1.63e-4` | `4.00` |
| 16 | `4.08e-5` | `4.00` |
| 32 | `1.02e-5` | `4.00` |

SIA produces the same midpoint errors for this Hermitian problem.
Crank--Nicolson also approaches a factor of four, with an additional Cayley
approximation error.

## Interpretation

These tests establish that:

- SIL evaluates a Hermitian exponential correctly;
- SIL evaluates both real- and imaginary-time Hermitian exponentials;
- SIA evaluates both Hermitian and CAP-like non-Hermitian exponentials;
- Crank--Nicolson converges to the same reference propagators;
- SIA retains physical absorber-induced norm loss;
- midpoint evaluation produces the intended second-order temporal behavior.

They do not establish an optimal Krylov dimension, tolerance, or time step for
a large TDSE calculation. Those parameters must be converged for the actual
Hamiltonian and compared using total operator calls and wall-clock cost.
