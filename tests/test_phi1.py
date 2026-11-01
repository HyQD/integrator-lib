"""Matrix-free Lanczos and Arnoldi actions of the first phi function."""

import numpy as np
import pytest
from scipy.linalg import expm

from integrator_lib import (
    ArnoldiPhi1,
    KrylovConvergenceError,
    LanczosPhi1,
)


HERMITIAN = np.array(
    [
        [0.7, 0.2j, -0.1, 0.05],
        [-0.2j, -0.4, 0.3, 0.1j],
        [-0.1, 0.3, 1.2, -0.15j],
        [0.05, -0.1j, 0.15j, 0.2],
    ],
    dtype=complex,
)
STATE = np.array([1.0 + 0.2j, -0.3 + 0.4j, 0.1 - 0.2j, 0.7])


def exact_phi1_action(generator, vector, step_size):
    """Dense augmented-exponential reference for h phi_1(h L) vector."""

    size = generator.shape[0]
    augmented = np.zeros((size + 1, size + 1), dtype=np.result_type(generator, vector))
    augmented[:size, :size] = step_size * generator
    augmented[:size, size] = vector
    return step_size * expm(augmented)[:size, size]


@pytest.mark.parametrize("reorthogonalize", [False, True])
def test_lanczos_phi1_matches_exact_hermitian_action(reorthogonalize):
    step_size = 0.7
    original = STATE.copy()
    action = LanczosPhi1(
        generator_factor=-1.0,
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
        reorthogonalize=reorthogonalize,
    )

    result = action(
        STATE,
        0.3,
        step_size,
        lambda values, time: -HERMITIAN @ values,
    )

    expected = exact_phi1_action(-HERMITIAN, STATE, step_size)
    np.testing.assert_allclose(result, expected, atol=3.0e-13, rtol=3.0e-13)
    np.testing.assert_array_equal(STATE, original)
    assert action.last_operator_calls == 4
    assert action.last_krylov_dimension == 4
    assert action.last_error_estimate == 0.0
    assert action.last_attempts[0].breakdown


def test_arnoldi_phi1_matches_exact_nonhermitian_action():
    cap = np.diag([0.0, 0.1, 0.4, 0.8])
    generator = -1j * HERMITIAN - cap
    step_size = 0.6
    action = ArnoldiPhi1(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )

    result = action(
        STATE,
        0.0,
        step_size,
        lambda values, time: generator @ values,
    )

    expected = exact_phi1_action(generator, STATE, step_size)
    np.testing.assert_allclose(result, expected, atol=3.0e-13, rtol=3.0e-13)
    assert action.last_attempts[0].breakdown


@pytest.mark.parametrize("factory", [LanczosPhi1, ArnoldiPhi1])
def test_phi1_residual_estimate_accepts_an_accurate_incomplete_space(factory):
    rng = np.random.default_rng(3214)
    matrix = rng.normal(size=(12, 12))
    hamiltonian = 0.5 * (matrix + matrix.T) + 3.0 * np.eye(12)
    state = rng.normal(size=12)
    step_size = 0.5
    options = {"generator_factor": -1.0} if factory is LanczosPhi1 else {}
    action = factory(
        max_krylov_dimension=8,
        rtol=1.0e-4,
        atol=0.0,
        **options,
    )

    result = action(
        state,
        0.0,
        step_size,
        lambda values, time: -hamiltonian @ values,
    )

    expected = exact_phi1_action(-hamiltonian, state, step_size)
    tolerance = action.rtol * max(
        abs(step_size) * np.linalg.norm(state),
        np.linalg.norm(result),
    )
    assert action.last_krylov_dimension < state.size
    assert np.linalg.norm(result - expected) <= tolerance


def test_lanczos_phi1_preserves_real_arithmetic():
    hamiltonian = np.array([[1.0, -0.2], [-0.2, 1.5]])
    state = np.array([1.0, -0.4])
    action = LanczosPhi1(
        generator_factor=-1.0,
        max_krylov_dimension=2,
        rtol=1.0e-13,
        atol=1.0e-15,
    )

    result = action(
        state,
        0.0,
        0.3,
        lambda values, time: -hamiltonian @ values,
    )

    assert not np.iscomplexobj(result)
    np.testing.assert_allclose(
        result,
        exact_phi1_action(-hamiltonian, state, 0.3),
        atol=1.0e-13,
    )


@pytest.mark.parametrize("factory", [LanczosPhi1, ArnoldiPhi1])
def test_phi1_actions_support_weighted_and_batched_inner_products(factory):
    weights = np.array([0.4, 1.1, 2.0])
    metric = np.diag(weights)
    metric_sqrt = np.diag(np.sqrt(weights))
    metric_inverse_sqrt = np.diag(1.0 / np.sqrt(weights))
    transformed = np.array(
        [[0.8, 0.2j, -0.1], [-0.2j, -0.3, 0.25], [-0.1, 0.25, 1.1]],
        dtype=complex,
    )
    hamiltonian = metric_inverse_sqrt @ transformed @ metric_sqrt
    state = np.array([1.0 + 0.1j, -0.2 + 0.3j, 0.5 - 0.4j])
    batch_sizes = []

    def inner_product(left, right):
        return np.vdot(left, metric @ right)

    def inner_products(left, right):
        batch_sizes.append(len(left))
        return np.asarray([inner_product(vector, right) for vector in left])

    options = {}
    if factory is LanczosPhi1:
        options["generator_factor"] = -1.0
    action = factory(
        max_krylov_dimension=3,
        rtol=1.0e-13,
        atol=1.0e-15,
        inner_product=inner_product,
        inner_products=inner_products,
        **options,
    )
    result = action(
        state,
        0.2,
        0.4,
        lambda values, time: -hamiltonian @ values,
    )

    expected = exact_phi1_action(-hamiltonian, state, 0.4)
    np.testing.assert_allclose(result, expected, atol=4.0e-13, rtol=4.0e-13)
    assert batch_sizes
    assert max(batch_sizes) > 1


@pytest.mark.parametrize("factory", [LanczosPhi1, ArnoldiPhi1])
def test_phi1_action_uses_midpoint_and_preserves_array_shape(factory):
    times = []
    state = STATE.reshape(2, 2)

    def rhs(values, time):
        times.append(time)
        return (-1j * HERMITIAN @ values.reshape(-1)).reshape(values.shape)

    result = factory(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )(state, 1.2, 0.4, rhs)

    expected = exact_phi1_action(-1j * HERMITIAN, state.reshape(-1), 0.4)
    np.testing.assert_allclose(
        result.reshape(-1), expected, atol=3.0e-13, rtol=3.0e-13
    )
    assert result.shape == state.shape
    assert times
    np.testing.assert_allclose(times, 1.4)


@pytest.mark.parametrize("factory", [LanczosPhi1, ArnoldiPhi1])
def test_phi1_zero_vector_and_zero_step(factory):
    action = factory()
    zero = np.zeros(3)
    result = action(zero, 0.0, 0.2, lambda values, time: values)
    np.testing.assert_array_equal(result, zero)
    assert action.last_operator_calls == 0
    assert action.last_accepted_substeps == 1

    state = np.arange(3.0)
    result = action(state, 0.0, 0.0, lambda values, time: values)
    np.testing.assert_array_equal(result, np.zeros_like(state))
    assert action.last_attempts == ()


def test_lanczos_phi1_rejects_nonhermitian_hamiltonian():
    hamiltonian = np.diag([1.0 + 0.2j, 0.5])
    action = LanczosPhi1(max_krylov_dimension=2)

    with pytest.raises(ValueError, match="Hermitian Hamiltonian"):
        action(
            np.array([1.0, 0.0]),
            0.0,
            0.1,
            lambda values, time: -1j * hamiltonian @ values,
        )


@pytest.mark.parametrize("factory", [LanczosPhi1, ArnoldiPhi1])
def test_phi1_raises_when_one_krylov_space_does_not_converge(factory):
    generator = -np.diag([0.1, 0.5, 1.2, 2.1])
    options = {"generator_factor": -1.0} if factory is LanczosPhi1 else {}
    action = factory(
        max_krylov_dimension=1,
        rtol=1.0e-14,
        **options,
    )

    with pytest.raises(KrylovConvergenceError, match="phi_1 action"):
        action(np.ones(4), 0.0, 1.0, lambda values, time: generator @ values)

    assert action.last_rejected_substeps == 1
    assert action.last_accepted_substeps == 0
    assert action.last_operator_calls == 1


@pytest.mark.parametrize("factory", [LanczosPhi1, ArnoldiPhi1])
def test_phi1_rejects_shape_changing_right_hand_side(factory):
    with pytest.raises(ValueError, match="preserve"):
        factory()(np.ones(2), 0.0, 0.1, lambda values, time: np.ones(3))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: LanczosPhi1(max_krylov_dimension=0), "dimension"),
        (lambda: ArnoldiPhi1(rtol=0.0, atol=0.0), "At least one"),
        (lambda: LanczosPhi1(inner_product=1), "inner_product"),
        (lambda: ArnoldiPhi1(inner_products=1), "inner_products"),
        (lambda: LanczosPhi1(generator_factor=0.0), "generator_factor"),
    ],
)
def test_invalid_phi1_configuration(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()
