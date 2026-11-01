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
- [Krylov exponential and phi-function actions](krylov_exponential.md)
- [Krylov exponential and phi-function validation](krylov_validation.md)

The Krylov page documents short-iterative Lanczos (SIL), short-iterative
Arnoldi (SIA), and matrix-free first-phi-function actions, including custom
weighted and distributed inner products. In the public Python API, SIA is
implemented by `ArnoldiExponential`; the phi-function actions are implemented
by `LanczosPhi1` and `ArnoldiPhi1`.

Runnable usage fragments are included in the feature pages. The test suite
contains small exact-matrix reference problems for every public propagator.
