# Fourth-order commutator-free Magnus integration

`CommutatorFreeMagnus4` advances a linear time-dependent equation

$$
\frac{d}{dt}y(t) = L(t)y(t)
$$

with a fourth-order commutator-free Magnus formula. It evaluates the generator
at the two fourth-order Gauss--Legendre nodes and applies two exponentials of
weighted generator combinations. The method requires neither explicit
matrices nor operator commutators.

## Basic usage

The class wraps an exponential action that follows the standard
`integrator-lib` call convention:

```python
from integrator_lib import CommutatorFreeMagnus4, ArnoldiExponential

integrator = CommutatorFreeMagnus4(
    ArnoldiExponential(
        max_krylov_dimension=20,
        rtol=1e-10,
    )
)

new_state = integrator(
    state,
    time,
    time_step,
    linear_rhs,
)
```

For a time-dependent Schrödinger equation, `linear_rhs` returns
`-1j * H(t) @ state`. Use `ShortIterativeLanczos` when every weighted
Hamiltonian combination is Hermitian under the configured inner product. Use
`ArnoldiExponential` for a general generator, including a Hamiltonian with a
complex absorbing potential.

## Formula and ordering

For a step of length `h`, define

$$
c_{1,2}=\frac{1}{2}\mp\frac{\sqrt{3}}{6},
\qquad
L_j=L(t+c_j h),
$$

and

$$
a_1=\frac{3-2\sqrt{3}}{12},
\qquad
a_2=\frac{3+2\sqrt{3}}{12}.
$$

The method applies the early-dominated exponential first:

$$
y_{n+1}=
\exp\!\left[h(a_1L_1+a_2L_2)\right]
\exp\!\left[h(a_2L_1+a_1L_2)\right]y_n.
$$

Because operators act from right to left, reversing the application order is
not equivalent when `L(t_1)` and `L(t_2)` do not commute. The test suite
checks that the documented ordering is fourth order and that its reversal is
only second order on a noncommuting problem.

## Structured operator combinations

The default implementation evaluates `linear_rhs` at both Gauss nodes for
every combined-operator action. Applications can avoid duplicated expensive
work by supplying `linear_combination_factory`:

```python
def combination_factory(linear_rhs, nodes, weights):
    effective_field = sum(
        weight * field(node)
        for node, weight in zip(nodes, weights, strict=True)
    )

    def action(state):
        return sum(weights) * field_free_action(state) + (
            effective_field * interaction_action(state)
        )

    return action


integrator = CommutatorFreeMagnus4(
    ArnoldiExponential(),
    linear_combination_factory=combination_factory,
)
```

The factory receives the original `linear_rhs`, the two absolute Gauss-node
times, and the weights for one exponential. It must return a one-argument
matrix-free action for the frozen weighted combination.

## Diagnostics

When the exponential action provides Krylov diagnostics, CF4 aggregates them
across both exponentials:

- operator, accepted-substep, and rejected-substep counts are summed;
- refinement depth and Krylov dimension are maximized;
- accepted error estimates are summed;
- Krylov attempts are retained in stage and application order.

The corresponding `last_stage_*` tuples retain each exponential's diagnostics
separately. Krylov tolerances control only the accuracy of each exponential
action; they do not estimate the outer fourth-order time-discretization error.
