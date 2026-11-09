"""Generic fourth-order commutator-free Magnus integration."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from integrator_lib import (
    ArnoldiExponential,
    CommutatorFreeMagnus4,
    KrylovConvergenceError,
    ShortIterativeLanczos,
)


class DenseExponential:
    """Exact exponential action for small one-dimensional test states."""

    def __call__(self, state, time, step_size, linear_rhs):
        dimension = state.size
        basis = np.eye(dimension, dtype=state.dtype)
        midpoint = time + 0.5 * step_size
        matrix = np.column_stack(
            [linear_rhs(vector, midpoint) for vector in basis]
        )
        return expm(step_size * matrix) @ state


def _matrices():
    constant = np.array([[0.0, 1.0], [-1.4, 0.2]])
    linear = np.array([[0.3, -0.8], [0.5, -0.4]])
    return constant, linear


def _propagate_cf4(n_steps):
    constant, linear = _matrices()
    final_time = 0.8
    step_size = final_time / n_steps
    state = np.array([1.0, -0.3])
    integrator = CommutatorFreeMagnus4(DenseExponential())
    rhs = lambda values, time: (constant + time * linear) @ values
    for step in range(n_steps):
        state = integrator(state, step * step_size, step_size, rhs)
    return state


def _reference_solution():
    constant, linear = _matrices()
    initial = np.array([1.0, -0.3])
    return solve_ivp(
        lambda time, values: (constant + time * linear) @ values,
        (0.0, 0.8),
        initial,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-15,
    ).y[:, -1]


def test_cf4_has_fourth_order_convergence_for_noncommuting_generator():
    reference = _reference_solution()
    errors = np.asarray(
        [np.linalg.norm(_propagate_cf4(steps) - reference) for steps in (2, 4, 8)]
    )

    ratios = errors[:-1] / errors[1:]
    assert np.all((15.5 < ratios) & (ratios < 17.0))


def test_cf4_uses_gauss_nodes_and_chronological_exponential_order():
    factory_calls = []

    def factory(linear_rhs, nodes, weights):
        factory_calls.append((linear_rhs, nodes, weights))
        return lambda values: np.zeros_like(values)

    rhs = lambda values, time: values
    integrator = CommutatorFreeMagnus4(
        DenseExponential(),
        linear_combination_factory=factory,
    )
    state = np.array([1.0, -0.4])
    result = integrator(state, 1.2, 0.3, rhs)

    expected_nodes = tuple(
        1.2 + 0.3 * node for node in integrator.gauss_nodes
    )
    assert len(factory_calls) == 2
    assert factory_calls[0][0] is rhs
    assert factory_calls[1][0] is rhs
    np.testing.assert_allclose(factory_calls[0][1], expected_nodes)
    np.testing.assert_allclose(factory_calls[1][1], expected_nodes)
    assert factory_calls[0][2] == integrator.stage_weights[0]
    assert factory_calls[1][2] == integrator.stage_weights[1]
    assert factory_calls[0][2][0] > factory_calls[0][2][1]
    assert factory_calls[1][2][0] < factory_calls[1][2][1]
    np.testing.assert_array_equal(result, state)


def test_reversing_cf4_exponential_order_is_only_second_order():
    constant, linear = _matrices()
    reference = _reference_solution()
    initial = np.array([1.0, -0.3])
    nodes = CommutatorFreeMagnus4.gauss_nodes
    reversed_weights = tuple(reversed(CommutatorFreeMagnus4.stage_weights))
    errors = []

    for n_steps in (4, 8, 16):
        step_size = 0.8 / n_steps
        state = initial.copy()
        for step in range(n_steps):
            time = step * step_size
            matrices = tuple(
                constant + (time + node * step_size) * linear for node in nodes
            )
            for weights in reversed_weights:
                generator = sum(
                    weight * matrix
                    for weight, matrix in zip(weights, matrices, strict=True)
                )
                state = expm(step_size * generator) @ state
        errors.append(np.linalg.norm(state - reference))

    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all((3.9 < ratios) & (ratios < 4.2))


def test_cf4_default_combination_samples_each_node_for_each_action():
    calls = []

    def rhs(values, time):
        calls.append(time)
        return time * values

    state = np.array([1.0])
    integrator = CommutatorFreeMagnus4(DenseExponential())
    integrator(state, 0.7, 0.2, rhs)

    expected_nodes = tuple(0.7 + 0.2 * node for node in integrator.gauss_nodes)
    assert calls == [
        expected_nodes[0],
        expected_nodes[1],
        expected_nodes[0],
        expected_nodes[1],
    ]


def test_cf4_custom_factory_can_avoid_original_rhs_evaluations():
    original_calls = []
    factory_calls = []

    def rhs(values, time):
        original_calls.append(time)
        return values

    def factory(linear_rhs, nodes, weights):
        factory_calls.append((linear_rhs, nodes, weights))
        return lambda values: sum(weights) * values

    state = np.array([1.0, -0.3])
    integrator = CommutatorFreeMagnus4(
        DenseExponential(),
        linear_combination_factory=factory,
    )
    result = integrator(state, 0.0, 0.4, rhs)

    np.testing.assert_allclose(result, np.exp(0.4) * state, atol=2.0e-15)
    assert len(factory_calls) == 2
    assert original_calls == []


@pytest.mark.parametrize("exponential", ["lanczos", "arnoldi"])
def test_cf4_supports_hermitian_krylov_exponential_actions(exponential):
    h0 = np.array([[0.7, 0.2j], [-0.2j, -0.3]])
    h1 = np.array([[0.1, -0.4], [-0.4, 0.5]])
    state = np.array([1.0 + 0.2j, -0.3 + 0.4j])
    rhs = lambda values, time: -1j * (h0 + time * h1) @ values
    dense = CommutatorFreeMagnus4(DenseExponential())
    if exponential == "lanczos":
        action = ShortIterativeLanczos(
            max_krylov_dimension=2,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    else:
        action = ArnoldiExponential(
            max_krylov_dimension=2,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    krylov = CommutatorFreeMagnus4(action)

    expected = dense(state, 0.2, 0.1, rhs)
    result = krylov(state, 0.2, 0.1, rhs)

    np.testing.assert_allclose(result, expected, atol=3.0e-13, rtol=3.0e-13)
    assert len(krylov.last_stage_attempts) == 2
    assert len(krylov.last_attempts) == sum(
        len(attempts) for attempts in krylov.last_stage_attempts
    )
    assert krylov.last_operator_calls == sum(krylov.last_stage_operator_calls)
    assert krylov.last_accepted_substeps == sum(
        krylov.last_stage_accepted_substeps
    )
    assert krylov.last_rejected_substeps == sum(
        krylov.last_stage_rejected_substeps
    )
    assert krylov.last_refinement_depth == max(
        krylov.last_stage_refinement_depths
    )
    assert krylov.last_krylov_dimension == max(
        krylov.last_stage_krylov_dimensions
    )
    assert krylov.last_error_estimate == pytest.approx(
        sum(krylov.last_stage_error_estimates)
    )


def test_cf4_supports_general_nonhermitian_arnoldi_action():
    generator0 = np.array([[0.1j, 0.4], [-0.2, -0.3 - 0.2j]])
    generator1 = np.array([[0.2, -0.1j], [0.3j, -0.4]])
    state = np.array([1.0 + 0.2j, -0.3 + 0.4j])
    rhs = lambda values, time: (generator0 + time * generator1) @ values
    dense = CommutatorFreeMagnus4(DenseExponential())
    krylov = CommutatorFreeMagnus4(
        ArnoldiExponential(
            max_krylov_dimension=2,
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    )

    expected = dense(state, 0.2, 0.1, rhs)
    result = krylov(state, 0.2, 0.1, rhs)

    np.testing.assert_allclose(result, expected, atol=3.0e-13, rtol=3.0e-13)


def test_cf4_zero_step_preserves_input_without_evaluating_callbacks():
    calls = []
    state = np.array([1.0, -0.4])
    original = state.copy()
    integrator = CommutatorFreeMagnus4(DenseExponential())

    result = integrator(
        state,
        0.0,
        0.0,
        lambda values, time: calls.append((values, time)),
    )

    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(state, original)
    assert result is not state
    assert calls == []
    assert integrator.last_stage_attempts == ()
    assert integrator.last_operator_calls == 0


def test_cf4_retains_diagnostics_from_a_failed_stage():
    hamiltonian = np.diag([0.1, 0.5, 1.2, 2.1])
    action = ShortIterativeLanczos(
        max_krylov_dimension=1,
        rtol=1.0e-14,
        max_refinements=0,
    )
    integrator = CommutatorFreeMagnus4(action)

    with pytest.raises(KrylovConvergenceError, match="Krylov exponential"):
        integrator(
            np.ones(4),
            0.0,
            1.0,
            lambda values, time: -1j * hamiltonian @ values,
        )

    assert len(integrator.last_stage_attempts) == 1
    assert integrator.last_operator_calls == 1
    assert integrator.last_accepted_substeps == 0
    assert integrator.last_rejected_substeps == 1


def test_cf4_validates_configuration_callbacks_and_factory_result():
    with pytest.raises(TypeError, match="exponential_integrator"):
        CommutatorFreeMagnus4(None)
    with pytest.raises(TypeError, match="linear_combination_factory"):
        CommutatorFreeMagnus4(DenseExponential(), linear_combination_factory=1)

    integrator = CommutatorFreeMagnus4(DenseExponential())
    with pytest.raises(TypeError, match="linear_rhs"):
        integrator(np.ones(2), 0.0, 0.1, None)

    invalid_factory = CommutatorFreeMagnus4(
        DenseExponential(),
        linear_combination_factory=lambda linear_rhs, nodes, weights: None,
    )
    with pytest.raises(TypeError, match="must return a callable"):
        invalid_factory(np.ones(2), 0.0, 0.1, lambda values, time: values)


def test_cf4_rejects_shape_changing_rhs():
    integrator = CommutatorFreeMagnus4(DenseExponential())
    with pytest.raises(ValueError, match="preserve"):
        integrator(
            np.ones(2),
            0.0,
            0.1,
            lambda values, time: np.ones(3),
        )
