# Exponential Euler

`ExponentialEuler` implements the first-order exponential-Euler update

$$
y_{n+1}
=
y_n+h\varphi_1(hL)F(y_n,t_n),
\qquad
\varphi_1(z)=\frac{e^z-1}{z}.
$$

Here $F$ is the complete rate and $L$ is the selected linear part whose stiff
action is treated through the phi function. For a semilinear equation

$$
y'=Ly+N(y,t),
$$

the complete rate is $F(y,t)=Ly+N(y,t)$. The equivalent conventional form is

$$
y_{n+1}=e^{hL}y_n+h\varphi_1(hL)N(y_n,t_n).
$$

The residual form is also useful for stationary-state calculations: the
caller may supply a complete stationary residual as $F$, so every zero of the
residual is a fixed point of the unnormalized update.

## Configuration

The stepper composes with a separately configured phi-function action. This
keeps the choice between a Hermitian Lanczos projection and a general Arnoldi
projection explicit:

```python
from integrator_lib import ExponentialEuler, LanczosPhi1

phi1_action = LanczosPhi1(
    generator_factor=-1.0,
    max_krylov_dimension=30,
    rtol=1.0e-10,
    atol=0.0,
    inner_product=representation.inner_product,
    inner_products=representation.inner_products,
)
integrator = ExponentialEuler(phi1_action)
```

One step has two operator callbacks:

```python
def full_rhs(state, time):
    return complete_rate(state, time)

def linear_rhs(state, time):
    return apply_selected_linear_generator(state, time)

new_state = integrator(
    state,
    time,
    time_step,
    full_rhs,
    linear_rhs,
)
```

`full_rhs` is evaluated exactly once at the beginning of every nonzero step.
`linear_rhs` is forwarded to the configured phi-function action. The current
Krylov phi-function actions sample it at the temporal midpoint. For an
autonomous operator this distinction disappears; for a time-dependent split,
the complete scheme remains first order and its time-step convergence must be
checked by the application.

The input state is not modified. A zero-length step returns a copy without
evaluating either callback.

## Division of responsibilities

`ExponentialEuler` does not:

- choose the split between $F$ and $L$;
- construct a Hamiltonian or stationary residual;
- normalize or orthogonalize the result;
- enforce symmetry constraints;
- select a Krylov tolerance or inner product;
- decide whether the iteration has converged.

Those decisions belong to the calling application. In particular, a
ground-state solver may normalize after each step and evaluate energy, state
change and eigenstate residual independently.

## Diagnostics

The most recent step records:

```python
integrator.last_rhs_evaluations
integrator.last_phi1_operator_calls
integrator.last_phi1_attempts
integrator.last_phi1_krylov_dimension
integrator.last_phi1_error_estimate
```

The phi-function diagnostics are copied from the configured action even when
that action raises `KrylovConvergenceError`. This lets an outer solver report
why a step failed. A custom phi-function callable without these attributes is
still supported; unavailable diagnostics are recorded as `None`.

## Validation

The tests compare the implementation with dense matrix exponentials for
constant affine equations and Hermitian imaginary-time evolution. They also
verify first-order convergence for a nonlinear rate, callback evaluation
counts and times, real arithmetic, shape preservation, zero-step behavior,
input preservation, validation failures and retained Krylov diagnostics.
