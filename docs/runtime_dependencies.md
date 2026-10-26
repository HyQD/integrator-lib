# Runtime dependencies

`integrator-lib` has two numerical runtime dependencies:

- [NumPy](https://numpy.org/) supplies array storage, scalar validation, and
  inner products.
- [SciPy](https://scipy.org/) supplies iterative linear solvers,
  `LinearOperator`, and dense exponentiation of the small projected Krylov
  matrices.

The package deliberately does not depend on spatial-grid or physics-model
libraries. In particular:

- FEM-DVR grids and differentiation operators belong to `grid-lib`;
- Hamiltonian assembly and wavefunction representations belong to packages
  such as `grid-tdse`;
- temporal electromagnetic-field profiles belong to
  `electromagnetic-fields`.

This separation keeps every integrator matrix-free and independent of the
coordinate representation. The caller supplies the complete operator action
and, where necessary, the representation-specific inner product or
preconditioner.

Supported Python, NumPy, and SciPy versions are declared in
[`pyproject.toml`](../pyproject.toml). That file is the authoritative runtime
dependency specification.
