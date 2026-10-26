# integrator-lib documentation

`integrator-lib` provides coordinate-independent integration algorithms for
array-valued differential equations. Spatial grids, Hamiltonians, physical
models, normalization policies, and observables belong to the calling
application.

The documentation is organized around the common integrator contract and the
algorithms currently intended for time-dependent quantum dynamics:

- [Installation](installation.md)
- [Runtime dependencies](runtime_dependencies.md)
- [Integrator interface and conventions](integrator_interface.md)
- [Crank--Nicolson propagation](crank_nicolson.md)
- [Short-iterative Lanczos and Arnoldi propagation](krylov_exponential.md)
- [SIL and SIA validation](krylov_validation.md)

The Krylov page documents both short-iterative Lanczos (SIL) and
short-iterative Arnoldi (SIA). In the public Python API, SIA is implemented by
`ArnoldiExponential`.

Runnable usage fragments are included in the feature pages. The test suite
contains small exact-matrix reference problems for every public propagator.
