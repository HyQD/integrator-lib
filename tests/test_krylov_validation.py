import numpy as np
import pytest
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


def _sil():
    return ShortIterativeLanczos(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def _sia():
    return ArnoldiExponential(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def _cn():
    return CrankNicolson(rtol=1.0e-13, atol=1.0e-15)


def _propagate(integrator, rhs, n_steps, final_time=1.0):
    step_size = final_time / n_steps
    state = INITIAL_STATE.copy()
    for step in range(n_steps):
        state = integrator(state, step * step_size, step_size, rhs)
    return state


def test_sil_and_sia_match_exact_constant_hermitian_propagation():
    final_time = 1.0
    exact = expm(-1j * final_time * HAMILTONIAN) @ INITIAL_STATE
    rhs = lambda state, time: -1j * HAMILTONIAN @ state

    for integrator in (_sil(), _sia()):
        state = _propagate(integrator, rhs, n_steps=5, final_time=final_time)
        np.testing.assert_allclose(state, exact, atol=2.0e-13, rtol=2.0e-13)
        assert np.vdot(state, state).real == pytest.approx(1.0, abs=2.0e-13)


def test_lanczos_and_arnoldi_match_exact_imaginary_time_propagation():
    final_imaginary_time = 0.7
    positive_hamiltonian = HAMILTONIAN + 2.0 * np.eye(HAMILTONIAN.shape[0])
    exact = (
        expm(-final_imaginary_time * positive_hamiltonian) @ INITIAL_STATE
    )
    rhs = lambda state, time: -positive_hamiltonian @ state
    lanczos = ShortIterativeLanczos(
        generator_factor=-1.0,
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )

    for integrator in (lanczos, _sia()):
        state = _propagate(
            integrator,
            rhs,
            n_steps=5,
            final_time=final_imaginary_time,
        )
        np.testing.assert_allclose(state, exact, atol=2.0e-13, rtol=2.0e-13)


def test_crank_nicolson_has_second_order_cayley_error_for_constant_hamiltonian():
    exact = expm(-1j * HAMILTONIAN) @ INITIAL_STATE
    rhs = lambda state, time: -1j * HAMILTONIAN @ state
    errors = []
    for n_steps in (5, 10, 20):
        state = _propagate(_cn(), rhs, n_steps)
        errors.append(np.linalg.norm(state - exact))
        assert np.vdot(state, state).real == pytest.approx(1.0, abs=2.0e-12)

    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all((3.9 < ratios) & (ratios < 4.1))


def test_sia_matches_exact_cap_propagation_and_norm_loss():
    generator = -1j * HAMILTONIAN - CAP
    exact = expm(generator) @ INITIAL_STATE
    exact_norm = np.vdot(exact, exact).real
    rhs = lambda state, time: generator @ state

    state = _propagate(_sia(), rhs, n_steps=5)

    np.testing.assert_allclose(state, exact, atol=2.0e-13, rtol=2.0e-13)
    assert np.vdot(state, state).real == pytest.approx(exact_norm, abs=2.0e-13)
    assert exact_norm < 1.0


def test_crank_nicolson_converges_to_exact_cap_propagation():
    generator = -1j * HAMILTONIAN - CAP
    exact = expm(generator) @ INITIAL_STATE
    rhs = lambda state, time: generator @ state
    errors = [
        np.linalg.norm(_propagate(_cn(), rhs, n_steps) - exact)
        for n_steps in (5, 10, 20)
    ]

    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all((3.9 < ratios) & (ratios < 4.1))


def _scale(time):
    return 1.0 + 0.3 * time + 0.2 * time**2


@pytest.mark.parametrize("factory", [_sil, _sia, _cn])
def test_midpoint_time_dependent_propagation_is_second_order(factory):
    final_time = 1.0
    integrated_scale = final_time + 0.15 * final_time**2
    integrated_scale += (0.2 / 3.0) * final_time**3
    exact = expm(-1j * integrated_scale * HAMILTONIAN) @ INITIAL_STATE
    rhs = lambda state, time: -1j * _scale(time) * HAMILTONIAN @ state
    errors = [
        np.linalg.norm(_propagate(factory(), rhs, n_steps) - exact)
        for n_steps in (4, 8, 16)
    ]

    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all((3.8 < ratios) & (ratios < 4.2))
