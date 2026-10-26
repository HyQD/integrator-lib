"""Validate SIL and SIA against exact propagation and Crank--Nicolson.

Run from the repository root with::

    python examples/validate_krylov_propagators.py

The deterministic four-state problems are deliberately small enough that the
reference propagators can be formed with ``scipy.linalg.expm``. They exercise
a constant Hermitian Hamiltonian, a CAP-like non-Hermitian generator, and a
commuting time-dependent Hamiltonian sampled at temporal midpoints.
"""

import numpy as np
from scipy.linalg import expm

from integrator_lib import (
    ArnoldiExponential,
    CrankNicolson,
    ShortIterativeLanczos,
)


HAMILTONIAN = np.array(
    [
        [0.7, 0.2j, -0.1, 0.05],
        [-0.2j, -0.4, 0.3, 0.1j],
        [-0.1, 0.3, 1.2, -0.15j],
        [0.05, -0.1j, 0.15j, 0.2],
    ],
    dtype=complex,
)
CAP = np.diag([0.0, 0.05, 0.2, 0.5])
INITIAL_STATE = np.array([1.0 + 0.2j, -0.3 + 0.4j, 0.1 - 0.2j, 0.7])
INITIAL_STATE /= np.linalg.norm(INITIAL_STATE)
FINAL_TIME = 1.0
STEP_COUNTS = (5, 10, 20)


def sil():
    return ShortIterativeLanczos(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def sia():
    return ArnoldiExponential(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def crank_nicolson():
    return CrankNicolson(rtol=1.0e-13, atol=1.0e-15)


def propagate(integrator, linear_rhs, n_steps):
    step_size = FINAL_TIME / n_steps
    state = INITIAL_STATE.copy()
    for step in range(n_steps):
        state = integrator(
            state,
            step * step_size,
            step_size,
            linear_rhs,
        )
    return state


def convergence_order(previous_error, error):
    return np.log2(previous_error / error)


def print_hermitian_validation():
    exact = expm(-1j * FINAL_TIME * HAMILTONIAN) @ INITIAL_STATE
    rhs = lambda state, time: -1j * HAMILTONIAN @ state
    methods = (
        ("SIL", sil),
        ("SIA", sia),
        ("Crank--Nicolson", crank_nicolson),
    )

    print("Constant Hermitian Hamiltonian")
    print("method              dt          state error       norm error")
    print("-" * 66)
    for name, factory in methods:
        for n_steps in STEP_COUNTS:
            state = propagate(factory(), rhs, n_steps)
            error = np.linalg.norm(state - exact)
            norm_error = abs(np.vdot(state, state).real - 1.0)
            print(
                f"{name:18s} {FINAL_TIME / n_steps:8.3f} "
                f"{error:18.8e} {norm_error:16.8e}"
            )
    print()


def print_cap_validation():
    generator = -1j * HAMILTONIAN - CAP
    exact = expm(FINAL_TIME * generator) @ INITIAL_STATE
    exact_norm = np.vdot(exact, exact).real
    rhs = lambda state, time: generator @ state
    methods = (
        ("SIA", sia),
        ("Crank--Nicolson", crank_nicolson),
    )

    print("CAP-like non-Hermitian generator")
    print("method              dt          state error        final norm")
    print("-" * 66)
    for name, factory in methods:
        for n_steps in STEP_COUNTS:
            state = propagate(factory(), rhs, n_steps)
            error = np.linalg.norm(state - exact)
            norm = np.vdot(state, state).real
            print(
                f"{name:18s} {FINAL_TIME / n_steps:8.3f} "
                f"{error:18.8e} {norm:16.8e}"
            )
    print(f"Exact final norm: {exact_norm:.8e}\n")


def time_dependent_scale(time):
    return 1.0 + 0.3 * time + 0.2 * time**2


def integrated_time_dependent_scale(final_time):
    return final_time + 0.15 * final_time**2 + (0.2 / 3.0) * final_time**3


def print_midpoint_validation():
    integral = integrated_time_dependent_scale(FINAL_TIME)
    exact = expm(-1j * integral * HAMILTONIAN) @ INITIAL_STATE
    rhs = lambda state, time: (
        -1j * time_dependent_scale(time) * HAMILTONIAN @ state
    )
    methods = (
        ("SIL", sil),
        ("SIA", sia),
        ("Crank--Nicolson", crank_nicolson),
    )
    step_counts = (4, 8, 16, 32)

    print("Quadratic time dependence with midpoint sampling")
    print("method            steps          state error    observed order")
    print("-" * 68)
    for name, factory in methods:
        previous_error = None
        for n_steps in step_counts:
            state = propagate(factory(), rhs, n_steps)
            error = np.linalg.norm(state - exact)
            order = (
                "-"
                if previous_error is None
                else f"{convergence_order(previous_error, error):.6f}"
            )
            print(f"{name:18s} {n_steps:5d} {error:20.8e} {order:>17s}")
            previous_error = error


if __name__ == "__main__":
    print_hermitian_validation()
    print_cap_validation()
    print_midpoint_validation()
