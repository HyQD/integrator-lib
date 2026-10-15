# integrator-lib

`integrator-lib` provides coordinate-independent time integration methods for
array-valued and compound states. The numerical methods accept right-hand-side
or step callables and do not depend on a particular spatial representation or
quantum-dynamics model.

The initial public API contains:

- `RK4`: classical four-stage, fourth-order Runge--Kutta integration.
- `CrankNicolson`: matrix-free implicit midpoint/Crank--Nicolson integration
  using SciPy GMRES or BiCGSTAB, with optional preconditioning.
- `Strang2`: second-order symmetric `A/2 - B - A/2` composition.
- `LinearSolveError`: raised when an iterative Crank--Nicolson solve fails.

## Installation

Install from a local checkout:

```bash
python -m pip install .
```

For editable development installation in the `scipy-1.18.0` conda environment:

```bash
conda activate scipy-1.18.0
python -m pip install --no-build-isolation -e ".[dev]"
```

## Usage

```python
from integrator_lib import CrankNicolson, RK4, Strang2

integrator = Strang2(
    linear_integrator=CrankNicolson(
        solver="gmres",
        rtol=1e-11,
        preconditioner=preconditioner,
    ),
    nonlinear_integrator=RK4(),
)

new_state = integrator(
    state,
    time,
    time_step,
    linear_rhs,
    nonlinear_rhs,
)
```

Every right-hand side returns a time derivative. For an equation written as
`i dy/dt = H(t)y`, the corresponding callable returns `-1j * H(t)y`.

`Strang2` treats its state as an opaque object. Its component steppers may
therefore evolve a NumPy array or a compound state containing several coupled
variables.

## Development

Run the tests with:

```bash
python -m pytest
```
