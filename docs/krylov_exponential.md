# Short-iterative Lanczos and Arnoldi propagation

`integrator-lib` provides two matrix-free exponential propagators:

| Method | Public class | Operator assumption | Typical use |
| --- | --- | --- | --- |
| Short-iterative Lanczos (SIL) | `ShortIterativeLanczos` | Hermitian Hamiltonian | Closed TDSE propagation without a CAP |
| Short-iterative Arnoldi (SIA) | `ArnoldiExponential` | General linear generator | TDSE propagation with a CAP or another non-Hermitian term |

Both methods construct a small Krylov projection using repeated calls to the
supplied operator. Only the projected matrix is exponentiated explicitly.
Neither method requires a preconditioner.

## Short-iterative Lanczos

For a Hermitian Hamiltonian, SIL constructs an orthonormal basis $V_m$ and a
real symmetric tridiagonal matrix $T_m$. The propagated state is approximated
by

$$
\Psi(t+\Delta t)
\approx
\lVert\Psi(t)\rVert V_m
\exp(-i\Delta t T_m)e_1.
$$

The public callable follows the common derivative convention and therefore
receives `-1j * H @ state`, not `H @ state` directly:

```python
from integrator_lib import ShortIterativeLanczos

integrator = ShortIterativeLanczos(
    max_krylov_dimension=20,
    rtol=1.0e-10,
    atol=0.0,
    max_refinements=8,
    reorthogonalize=True,
    inner_product=representation.inner_product,
)

def linear_rhs(values, evaluation_time):
    return -1j * apply_hamiltonian(values, evaluation_time)

new_state = integrator(
    state,
    time,
    time_step,
    linear_rhs,
)
```

The implementation recovers the Hamiltonian action as `1j * linear_rhs` and
checks that the Lanczos diagonal is real to numerical precision under the
configured inner product.

SIL must not be used when a CAP or another non-Hermitian contribution is
present. Use SIA instead.

## Short-iterative Arnoldi

SIA constructs an orthonormal basis $V_m$ and an upper-Hessenberg projection
$H_m$ for a general generator $L$. Its approximation is

$$
\Psi(t+\Delta t)
\approx
\lVert\Psi(t)\rVert V_m
\exp(\Delta t H_m)e_1.
$$

The implementation is exposed as `ArnoldiExponential`:

```python
from integrator_lib import ArnoldiExponential

integrator = ArnoldiExponential(
    max_krylov_dimension=30,
    rtol=1.0e-10,
    atol=0.0,
    max_refinements=8,
    reorthogonalize=True,
    inner_product=representation.inner_product,
)

new_state = integrator(state, time, time_step, linear_rhs)
```

For an effective CAP Hamiltonian

$$
H_{\mathrm{eff}}=H-iW,
$$

the supplied generator is

$$
L=-iH_{\mathrm{eff}}=-iH-W.
$$

The non-Hermitian Arnoldi projection permits the physical norm loss produced
by $W$.

## Weighted inner products

The default inner product is `numpy.vdot`. A discretized wavefunction may
instead require quadrature weights or a nontrivial metric. Supply a callable:

```python
integrator = ShortIterativeLanczos(
    inner_product=representation.inner_product,
)
```

The callable must accept two state-shaped arrays and return a finite scalar.
Self-inner-products must be real to numerical precision and nonnegative. The
Krylov vectors are orthonormalized using this same inner product.

## Reorthogonalization

Set `reorthogonalize=True` to improve numerical orthogonality:

- SIL applies a full correction against the stored Lanczos basis;
- SIA performs a second modified Gram--Schmidt pass.

Disabling reorthogonalization reduces inner-product work but can allow loss of
orthogonality for large Krylov spaces or difficult spectra. Accuracy and cost
should be measured for the intended Hamiltonian.

## Convergence estimate and refinement

After each Krylov extension, the implementation estimates the endpoint error
from the coupling to the next Krylov vector and the last component of the
projected exponential. The approximation is accepted when this estimate is no
larger than the allocated absolute-plus-relative tolerance.

If `max_krylov_dimension` is reached without convergence, the requested
interval is bisected recursively. Each refined substep samples the operator
at its own midpoint. The original interval's error budget is divided between
the accepted substeps.

```python
integrator = ArnoldiExponential(
    max_krylov_dimension=24,
    rtol=1.0e-10,
    atol=1.0e-12,
    max_refinements=6,
)
```

Exhausting the refinement depth raises `KrylovConvergenceError`.

Refinement controls the Krylov projection error. It is not an adaptive
estimate of the temporal midpoint approximation for a time-dependent
Hamiltonian. Time-step convergence must still be tested independently.

## Diagnostics

Every attempted substep is stored as a `KrylovAttempt` in:

```python
integrator.last_attempts
```

Each record contains:

- start time and step size;
- refinement depth;
- realized Krylov dimension;
- residual-based error estimate and allocated tolerance;
- number of operator calls;
- convergence and invariant-subspace-breakdown flags.

Aggregate diagnostics are available through:

```python
integrator.last_operator_calls
integrator.last_accepted_substeps
integrator.last_rejected_substeps
integrator.last_refinement_depth
integrator.last_krylov_dimension
integrator.last_error_estimate
```

`last_error_estimate` is the sum of the residual estimates for all accepted
substeps. Diagnostics are retained when propagation raises a convergence
error.

## Choosing SIL, SIA, or Crank--Nicolson

- Use SIL when the Hamiltonian is Hermitian and matrix-free Hamiltonian
  applications are efficient.
- Use SIA when a CAP or another contribution makes the generator
  non-Hermitian.
- Use Crank--Nicolson when its linear system can be solved efficiently with a
  suitable preconditioner or when a Cayley step is specifically desired.

The methods should be compared using total accepted-step cost, operator-call
count, memory use, norm or absorber behavior, and observable convergence. A
smaller Krylov dimension or iteration count alone does not determine the most
efficient propagator.

The deterministic exact-matrix comparisons are documented in
[SIL and SIA validation](krylov_validation.md).
