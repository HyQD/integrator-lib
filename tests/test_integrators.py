from dataclasses import dataclass

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

from integrator_lib import CNCMF1, CNCMF2, CrankNicolson, RK4, Strang2


def test_rk4_time_dependent_rhs_and_input_preservation():
    state = np.array([1.0, -2.0])
    original = state.copy()
    result = RK4()(state, 0.3, 0.4, lambda y, t: np.full_like(y, t))
    expected_increment = 0.5 * ((0.7) ** 2 - (0.3) ** 2)
    np.testing.assert_allclose(result, state + expected_increment, atol=1e-15)
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("solver", ["gmres", "bicgstab"])
def test_crank_nicolson_matches_cayley_step_and_preserves_norm(solver):
    hamiltonian = np.array([[0.8, 0.2j], [-0.2j, -0.3]])
    state = np.array([1.0 + 0.4j, -0.2 + 0.7j])
    dt = 0.17
    integrator = CrankNicolson(solver=solver, rtol=1e-13, atol=1e-15)
    result = integrator(
        state,
        0.5,
        dt,
        lambda y, t: -1j * hamiltonian @ y,
    )
    identity = np.eye(2)
    expected = np.linalg.solve(
        identity + 0.5j * dt * hamiltonian,
        (identity - 0.5j * dt * hamiltonian) @ state,
    )
    np.testing.assert_allclose(result, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(np.vdot(result, result), np.vdot(state, state), atol=2e-13)
    assert integrator.last_iterations > 0
    assert integrator.last_residual_norm < 1e-12


def test_crank_nicolson_evaluates_at_the_interval_midpoint():
    times = []

    def linear_rhs(state, time):
        times.append(time)
        return -1j * time * state

    CrankNicolson(rtol=1e-13)(np.ones(3), 1.2, 0.4, linear_rhs)
    np.testing.assert_allclose(times, 1.4)


def test_crank_nicolson_accepts_shaped_preconditioner():
    shape = (2, 3)
    scale = 0.7
    dt = 0.3
    calls = []

    def preconditioner(state):
        calls.append(state.shape)
        return state / (1 + 0.5j * dt * scale)

    integrator = CrankNicolson(
        preconditioner=preconditioner, rtol=1e-13, atol=1e-15
    )
    state = np.arange(6).reshape(shape)
    result = integrator(state, 0.0, dt, lambda y, t: -1j * scale * y)
    expected = (1 - 0.5j * dt * scale) / (1 + 0.5j * dt * scale) * state
    np.testing.assert_allclose(result, expected, atol=2e-13)
    assert calls
    assert set(calls) == {shape}


def test_crank_nicolson_checks_linear_operator_preconditioner_size():
    preconditioner = LinearOperator((3, 3), matvec=lambda x: x)
    integrator = CrankNicolson(preconditioner=preconditioner)
    with pytest.raises(ValueError, match="shape"):
        integrator(np.ones(2), 0.0, 0.1, lambda y, t: y)


@dataclass(frozen=True)
class TraceState:
    calls: tuple = ()


class RecordingStep:
    def __init__(self, name):
        self.name = name

    def __call__(self, state, time, step_size, rhs):
        return TraceState(state.calls + ((self.name, time, step_size, rhs),))


def test_strang2_is_opaque_to_state_and_uses_aba_intervals():
    linear_rhs = lambda state, time: state
    nonlinear_rhs = lambda state, time: state
    integrator = Strang2(RecordingStep("A"), RecordingStep("B"))
    result = integrator(TraceState(), 2.0, 0.6, linear_rhs, nonlinear_rhs)
    assert result.calls == (
        ("A", 2.0, 0.3, linear_rhs),
        ("B", 2.0, 0.6, nonlinear_rhs),
        ("A", 2.3, 0.3, linear_rhs),
    )


def _propagate_strang(n_steps):
    matrix_a = np.array([[0.0, 1.0], [-1.0, 0.0]])
    matrix_b = np.array([[0.0, 0.4], [0.7, 0.0]])
    state = np.array([1.0, -0.3])
    initial = state.copy()
    dt = 0.4 / n_steps
    integrator = Strang2(
        CrankNicolson(rtol=1e-13, atol=1e-15),
        RK4(),
    )
    for step in range(n_steps):
        state = integrator(
            state,
            step * dt,
            dt,
            lambda y, t: matrix_a @ y,
            lambda y, t: matrix_b @ y,
        )
    exact = expm(0.4 * (matrix_a + matrix_b)) @ initial
    return np.linalg.norm(state - exact)


def test_strang2_has_second_order_convergence_for_noncommuting_parts():
    errors = np.array([_propagate_strang(n) for n in (2, 4, 8)])
    ratios = errors[:-1] / errors[1:]
    assert np.all((3.8 < ratios) & (ratios < 4.2))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: CrankNicolson(solver="invalid"), "solver"),
        (lambda: CrankNicolson(rtol=0, atol=0), "At least one"),
        (lambda: Strang2(None, RK4()), "linear_integrator"),
        (lambda: Strang2(RK4(), None), "nonlinear_integrator"),
    ],
)
def test_invalid_integrator_configuration(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_integrators_reject_shape_changing_right_hand_sides():
    with pytest.raises(ValueError, match="preserve"):
        RK4()(np.ones(2), 0.0, 0.1, lambda y, t: np.ones(3))
    with pytest.raises(ValueError, match="preserve"):
        CrankNicolson()(np.ones(2), 0.0, 0.1, lambda y, t: np.ones(3))


def test_cncmf1_freezes_once_and_matches_a_frozen_cayley_step():
    state = np.array([1.0 + 0.2j, -0.3 + 0.4j])
    original = state.copy()
    matrix = np.array([[0.4, 0.1j], [-0.1j, -0.2]])
    references = []
    dt = 0.13
    time = 0.7

    def freeze(reference):
        references.append(reference.copy())
        scale = 1 + 0.2 * reference[0].real
        return lambda trial, evaluation_time: -1j * (
            matrix + scale * evaluation_time * np.eye(2)
        ) @ trial

    crank_nicolson = CrankNicolson(rtol=1e-13, atol=1e-15)
    integrator = CNCMF1(crank_nicolson)
    result = integrator(state, time, dt, freeze)

    midpoint = time + dt / 2
    scale = 1 + 0.2 * state[0].real
    frozen_matrix = matrix + scale * midpoint * np.eye(2)
    identity = np.eye(2)
    expected = np.linalg.solve(
        identity + 0.5j * dt * frozen_matrix,
        (identity - 0.5j * dt * frozen_matrix) @ state,
    )
    np.testing.assert_allclose(result, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(state, original)
    assert len(references) == 1
    np.testing.assert_array_equal(references[0], state)
    assert integrator.last_iterations == crank_nicolson.last_iterations
    assert integrator.last_residual_norm == crank_nicolson.last_residual_norm


class RecordingLinearStep:
    def __init__(self):
        self.calls = []
        self.last_iterations = None
        self.last_residual_norm = None

    def __call__(self, state, time, step_size, rhs):
        self.calls.append((time, step_size, rhs))
        self.last_iterations = len(self.calls)
        self.last_residual_norm = 10.0 ** (-len(self.calls))
        return state + step_size * rhs(state, time + step_size / 2)


@pytest.mark.parametrize(
    ("predictor_steps", "expected_calls"),
    [
        (1, [(2.0, 0.4)]),
        (2, [(2.0, 0.2), (2.2, 0.2)]),
    ],
)
def test_cncmf2_predictor_steps_and_frozen_midpoint_corrector(
    predictor_steps, expected_calls
):
    predictor = RecordingLinearStep()
    corrector = RecordingLinearStep()
    references = []

    def freeze(reference):
        references.append(reference.copy())
        coefficient = reference[0]
        return lambda trial, time: coefficient * trial

    state = np.array([1.0])
    original = state.copy()
    integrator = CNCMF2(
        predictor,
        corrector,
        predictor_steps=predictor_steps,
    )
    integrator(state, 2.0, 0.8, freeze)

    assert [(t, h) for t, h, _ in predictor.calls] == expected_calls
    assert [(t, h) for t, h, _ in corrector.calls] == [
        (2.0, 0.4),
        (2.4, 0.4),
    ]
    assert len(references) == 2
    np.testing.assert_array_equal(references[0], state)
    expected_midpoint = (1 + 0.4 / predictor_steps) ** predictor_steps
    np.testing.assert_allclose(references[1], [expected_midpoint])
    np.testing.assert_array_equal(state, original)
    assert integrator.last_predictor_iterations == tuple(
        range(1, predictor_steps + 1)
    )
    assert integrator.last_corrector_iterations == (1, 2)
    assert len(integrator.last_predictor_residual_norms) == predictor_steps
    assert len(integrator.last_corrector_residual_norms) == 2


def _propagate_cncmf2(n_steps, predictor_steps=2):
    state = np.array([1.0])
    dt = 0.2 / n_steps
    integrator = CNCMF2(
        CrankNicolson(rtol=1e-13, atol=1e-15),
        CrankNicolson(rtol=1e-13, atol=1e-15),
        predictor_steps=predictor_steps,
    )

    def freeze(reference):
        coefficient = reference[0]
        return lambda trial, time: coefficient * trial

    for step in range(n_steps):
        state = integrator(state, step * dt, dt, freeze)
    return abs(state[0] - 1.25)


def test_cncmf2_has_second_order_convergence_for_state_dependent_operator():
    errors = np.array([_propagate_cncmf2(n) for n in (4, 8, 16)])
    ratios = errors[:-1] / errors[1:]
    assert np.all((3.7 < ratios) & (ratios < 4.3))


@pytest.mark.parametrize("predictor_steps", [0, 3])
def test_cncmf2_rejects_unsupported_predictor_step_count(predictor_steps):
    with pytest.raises(ValueError, match="predictor_steps"):
        CNCMF2(RK4(), RK4(), predictor_steps=predictor_steps)


@pytest.mark.parametrize("predictor_steps", [True, 1.0, "2"])
def test_cncmf2_rejects_noninteger_predictor_step_count(predictor_steps):
    with pytest.raises(TypeError, match="predictor_steps"):
        CNCMF2(RK4(), RK4(), predictor_steps=predictor_steps)


def test_cncmf_integrators_validate_freeze_callback():
    with pytest.raises(TypeError, match="freeze"):
        CNCMF1(RK4())(np.ones(2), 0.0, 0.1, None)
    with pytest.raises(TypeError, match="return a callable"):
        CNCMF2(RK4(), RK4())(np.ones(2), 0.0, 0.1, lambda state: state)
