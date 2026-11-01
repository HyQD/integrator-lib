"""Generic exponential-Euler integration with matrix-free phi actions."""

import numpy as np
import pytest
from scipy.linalg import expm

from integrator_lib import (
    ArnoldiPhi1,
    ExponentialEuler,
    KrylovConvergenceError,
    LanczosPhi1,
)


def exact_affine_step(generator, state, forcing, step_size):
    """Exact solution of y' = generator y + forcing over one interval."""

    size = generator.shape[0]
    augmented = np.zeros(
        (size + 1, size + 1),
        dtype=np.result_type(generator, state, forcing),
    )
    augmented[:size, :size] = generator
    augmented[:size, size] = forcing
    initial = np.concatenate((state, [1]))
    return (expm(step_size * augmented) @ initial)[:size]


def test_exponential_euler_is_exact_for_a_constant_affine_equation():
    generator = np.array([[-0.4, 0.3], [-0.1, -0.8]])
    forcing = np.array([0.2, -0.05])
    state = np.array([1.0, -0.4])
    original = state.copy()
    step_size = 0.6
    stepper = ExponentialEuler(
        ArnoldiPhi1(
            max_krylov_dimension=2,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    )

    result = stepper(
        state,
        0.3,
        step_size,
        lambda values, time: generator @ values + forcing,
        lambda values, time: generator @ values,
    )

    expected = exact_affine_step(generator, state, forcing, step_size)
    np.testing.assert_allclose(result, expected, atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_array_equal(state, original)
    assert stepper.last_rhs_evaluations == 1
    assert stepper.last_phi1_operator_calls == 2
    assert stepper.last_phi1_krylov_dimension == 2


def test_exponential_euler_with_lanczos_reproduces_a_hermitian_exponential():
    hamiltonian = np.array([[1.0, -0.2], [-0.2, 1.5]])
    state = np.array([1.0, -0.4])
    step_size = 0.3
    stepper = ExponentialEuler(
        LanczosPhi1(
            generator_factor=-1.0,
            max_krylov_dimension=2,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    )

    result = stepper(
        state,
        0.0,
        step_size,
        lambda values, time: -hamiltonian @ values,
        lambda values, time: -hamiltonian @ values,
    )

    expected = expm(-step_size * hamiltonian) @ state
    assert not np.iscomplexobj(result)
    np.testing.assert_allclose(result, expected, atol=1.0e-13)


def test_exponential_euler_applies_the_complete_nonlinear_rate_once():
    linear_rate = -2.0
    state = np.array([0.3])
    step_size = 0.2
    rhs_calls = []
    stepper = ExponentialEuler(
        ArnoldiPhi1(
            max_krylov_dimension=1,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    )

    def full_rhs(values, time):
        rhs_calls.append((values.copy(), time))
        return linear_rate * values + values**2

    result = stepper(
        state,
        1.4,
        step_size,
        full_rhs,
        lambda values, time: linear_rate * values,
    )

    phi1 = np.expm1(step_size * linear_rate) / (step_size * linear_rate)
    expected = state + step_size * phi1 * (
        linear_rate * state + state**2
    )
    np.testing.assert_allclose(result, expected, atol=1.0e-14)
    assert len(rhs_calls) == 1
    np.testing.assert_array_equal(rhs_calls[0][0], state)
    assert rhs_calls[0][1] == 1.4


def test_exponential_euler_converges_at_first_order_for_a_nonlinear_rate():
    linear_rate = -2.0
    initial = 0.3
    final_time = 0.8
    exact = 1.0 / (
        (1.0 / initial - 0.5) * np.exp(2.0 * final_time) + 0.5
    )
    errors = []

    for n_steps in (20, 40, 80):
        step_size = final_time / n_steps
        state = np.array([initial])
        stepper = ExponentialEuler(
            ArnoldiPhi1(
                max_krylov_dimension=1,
                rtol=1.0e-13,
                atol=1.0e-15,
            )
        )
        for step in range(n_steps):
            state = stepper(
                state,
                step * step_size,
                step_size,
                lambda values, time: linear_rate * values + values**2,
                lambda values, time: linear_rate * values,
            )
        errors.append(abs(state[0] - exact))

    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    np.testing.assert_allclose(ratios, 2.0, rtol=0.06)


def test_exponential_euler_forwards_midpoint_sampling_to_the_phi1_action():
    full_times = []
    linear_times = []
    state = np.array([[1.0, -0.4], [0.2, 0.3]])
    diagonal = np.arange(1.0, 5.0).reshape(state.shape)
    stepper = ExponentialEuler(
        ArnoldiPhi1(
            max_krylov_dimension=4,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    )

    def full_rhs(values, time):
        full_times.append(time)
        return -diagonal * values

    def linear_rhs(values, time):
        linear_times.append(time)
        return -diagonal * values

    result = stepper(state, 1.2, 0.4, full_rhs, linear_rhs)

    np.testing.assert_allclose(result, np.exp(-0.4 * diagonal) * state)
    assert result.shape == state.shape
    assert full_times == [1.2]
    assert linear_times
    np.testing.assert_allclose(linear_times, 1.4)


def test_exponential_euler_zero_step_does_not_evaluate_callbacks():
    calls = []
    state = np.array([1.0, -0.4])
    stepper = ExponentialEuler(ArnoldiPhi1())

    result = stepper(
        state,
        0.0,
        0.0,
        lambda values, time: calls.append("full"),
        lambda values, time: calls.append("linear"),
    )

    np.testing.assert_array_equal(result, state)
    assert result is not state
    assert calls == []
    assert stepper.last_rhs_evaluations == 0
    assert stepper.last_phi1_operator_calls == 0
    assert stepper.last_phi1_attempts == ()


def test_exponential_euler_retains_failed_phi1_diagnostics():
    hamiltonian = np.diag([0.1, 0.5, 1.2, 2.1])
    stepper = ExponentialEuler(
        LanczosPhi1(
            generator_factor=-1.0,
            max_krylov_dimension=1,
            rtol=1.0e-14,
        )
    )

    with pytest.raises(KrylovConvergenceError, match="phi_1 action"):
        stepper(
            np.ones(4),
            0.0,
            1.0,
            lambda values, time: -hamiltonian @ values,
            lambda values, time: -hamiltonian @ values,
        )

    assert stepper.last_rhs_evaluations == 1
    assert stepper.last_phi1_operator_calls == 1
    assert len(stepper.last_phi1_attempts) == 1
    assert not stepper.last_phi1_attempts[0].converged


def test_exponential_euler_validates_configuration_and_callbacks():
    with pytest.raises(TypeError, match="phi1_action"):
        ExponentialEuler(None)

    stepper = ExponentialEuler(ArnoldiPhi1())
    with pytest.raises(TypeError, match="full_rhs"):
        stepper(np.ones(2), 0.0, 0.1, None, lambda values, time: values)
    with pytest.raises(TypeError, match="linear_rhs"):
        stepper(np.ones(2), 0.0, 0.1, lambda values, time: values, None)


def test_exponential_euler_rejects_invalid_full_rate_and_phi1_result():
    stepper = ExponentialEuler(ArnoldiPhi1())
    with pytest.raises(ValueError, match="preserve"):
        stepper(
            np.ones(2),
            0.0,
            0.1,
            lambda values, time: np.ones(3),
            lambda values, time: values,
        )

    class BadPhi1:
        def __call__(self, vector, time, step_size, linear_rhs):
            return np.ones(vector.size + 1)

    with pytest.raises(ValueError, match="preserve"):
        ExponentialEuler(BadPhi1())(
            np.ones(2),
            0.0,
            0.1,
            lambda values, time: values,
            lambda values, time: values,
        )
