import numpy as np
import pytest
from scipy.linalg import expm

from integrator_lib import (
    ArnoldiExponential,
    KrylovConvergenceError,
    ShortIterativeLanczos,
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
POSITIVE_HERMITIAN = HERMITIAN + 2.0 * np.eye(HERMITIAN.shape[0])


@pytest.mark.parametrize("reorthogonalize", [False, True])
def test_short_iterative_lanczos_matches_exact_hermitian_exponential(
    reorthogonalize,
):
    time = 0.3
    step_size = 0.7
    original = STATE.copy()
    integrator = ShortIterativeLanczos(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
        reorthogonalize=reorthogonalize,
    )

    result = integrator(
        STATE,
        time,
        step_size,
        lambda values, evaluation_time: -1j * HERMITIAN @ values,
    )

    expected = expm(-1j * step_size * HERMITIAN) @ STATE
    np.testing.assert_allclose(result, expected, atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_array_equal(STATE, original)
    assert integrator.last_accepted_substeps == 1
    assert integrator.last_rejected_substeps == 0
    assert integrator.last_operator_calls == 4
    assert integrator.last_krylov_dimension == 4
    assert integrator.last_error_estimate == 0.0
    assert integrator.last_attempts[0].breakdown


def test_arnoldi_matches_exact_nonhermitian_exponential():
    cap = np.diag([0.0, 0.1, 0.4, 0.8])
    generator = -1j * HERMITIAN - cap
    step_size = 0.6
    integrator = ArnoldiExponential(
        max_krylov_dimension=4,
        rtol=1.0e-13,
        atol=1.0e-15,
    )

    result = integrator(
        STATE,
        0.0,
        step_size,
        lambda values, time: generator @ values,
    )

    expected = expm(step_size * generator) @ STATE
    np.testing.assert_allclose(result, expected, atol=2.0e-13, rtol=2.0e-13)
    assert np.linalg.norm(result) < np.linalg.norm(STATE)
    assert integrator.last_attempts[0].breakdown


@pytest.mark.parametrize(
    "integrator",
    [
        ShortIterativeLanczos(
            generator_factor=-1.0,
            max_krylov_dimension=4,
            rtol=1.0e-13,
            atol=1.0e-15,
        ),
        ArnoldiExponential(
            max_krylov_dimension=4,
            rtol=1.0e-13,
            atol=1.0e-15,
        ),
    ],
)
def test_krylov_propagators_match_exact_imaginary_time_exponential(integrator):
    step_size = 0.6
    rhs = lambda values, time: -POSITIVE_HERMITIAN @ values

    result = integrator(STATE, 0.0, step_size, rhs)

    expected = expm(-step_size * POSITIVE_HERMITIAN) @ STATE
    np.testing.assert_allclose(result, expected, atol=2.0e-13, rtol=2.0e-13)
    assert np.linalg.norm(result) < np.linalg.norm(STATE)
    assert integrator.last_attempts[0].breakdown


def test_imaginary_time_lanczos_preserves_real_arithmetic():
    hamiltonian = np.array([[1.0, -0.2], [-0.2, 1.5]])
    state = np.array([1.0, -0.4])
    integrator = ShortIterativeLanczos(
        generator_factor=-1.0,
        max_krylov_dimension=2,
        rtol=1.0e-13,
        atol=1.0e-15,
    )

    result = integrator(
        state,
        0.0,
        0.3,
        lambda values, time: -hamiltonian @ values,
    )

    assert not np.iscomplexobj(result)
    np.testing.assert_allclose(
        result, expm(-0.3 * hamiltonian) @ state, atol=1.0e-13
    )


@pytest.mark.parametrize("factory", [ShortIterativeLanczos, ArnoldiExponential])
def test_krylov_propagators_support_weighted_inner_products(factory):
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

    integrator = factory(
        max_krylov_dimension=3,
        rtol=1.0e-13,
        atol=1.0e-15,
        inner_product=lambda left, right: np.vdot(left, metric @ right),
    )
    result = integrator(
        state,
        0.2,
        0.4,
        lambda values, time: -1j * hamiltonian @ values,
    )

    expected = expm(-0.4j * hamiltonian) @ state
    np.testing.assert_allclose(result, expected, atol=3.0e-13, rtol=3.0e-13)


@pytest.mark.parametrize(
    "factory",
    [ShortIterativeLanczos, ArnoldiExponential],
)
def test_krylov_propagators_use_midpoint_and_preserve_array_shape(factory):
    times = []
    state = STATE.reshape(2, 2)

    def rhs(values, time):
        times.append(time)
        flattened = values.reshape(-1)
        return (-1j * HERMITIAN @ flattened).reshape(values.shape)

    result = factory(
        max_krylov_dimension=4, rtol=1.0e-13, atol=1.0e-15
    )(state, 1.2, 0.4, rhs)

    expected = (expm(-0.4j * HERMITIAN) @ state.reshape(-1)).reshape(2, 2)
    np.testing.assert_allclose(result, expected, atol=2.0e-13, rtol=2.0e-13)
    assert result.shape == state.shape
    assert times
    np.testing.assert_allclose(times, 1.4)


@pytest.mark.parametrize("factory", [ShortIterativeLanczos, ArnoldiExponential])
def test_krylov_step_refinement_meets_a_small_subspace_tolerance(factory):
    size = 8
    hamiltonian = np.diag(np.linspace(-1.0, 3.0, size))
    hamiltonian += 0.1 * np.ones((size, size))
    state = np.linspace(0.2, 1.0, size).astype(complex)
    rhs = lambda values, time: -1j * hamiltonian @ values
    integrator = factory(
        max_krylov_dimension=4,
        rtol=1.0e-3,
        max_refinements=6,
    )

    result = integrator(state, 0.0, 1.0, rhs)

    expected = expm(-1j * hamiltonian) @ state
    np.testing.assert_allclose(result, expected, atol=1.2e-4, rtol=1.2e-4)
    assert integrator.last_accepted_substeps > 1
    assert integrator.last_rejected_substeps > 0
    assert integrator.last_refinement_depth > 0
    assert integrator.last_operator_calls == sum(
        attempt.operator_calls for attempt in integrator.last_attempts
    )
    assert integrator.last_error_estimate <= 1.0e-3 * np.linalg.norm(state)


@pytest.mark.parametrize("factory", [ShortIterativeLanczos, ArnoldiExponential])
def test_krylov_raises_after_exhausting_step_refinements(factory):
    diagonal = np.diag([0.1, 0.5, 1.2, 2.1])
    integrator = factory(
        max_krylov_dimension=1,
        rtol=1.0e-14,
        max_refinements=1,
    )
    rhs = lambda values, time: -1j * diagonal @ values

    with pytest.raises(KrylovConvergenceError, match="did not converge"):
        integrator(np.ones(4), 0.0, 1.0, rhs)

    assert integrator.last_rejected_substeps == 2
    assert integrator.last_accepted_substeps == 0
    assert integrator.last_refinement_depth == 1


def test_short_iterative_lanczos_rejects_nonhermitian_hamiltonian():
    hamiltonian = np.diag([1.0 + 0.2j, 0.5])
    integrator = ShortIterativeLanczos(max_krylov_dimension=2)

    with pytest.raises(ValueError, match="Hermitian Hamiltonian"):
        integrator(
            np.array([1.0, 0.0]),
            0.0,
            0.1,
            lambda values, time: -1j * hamiltonian @ values,
        )


def test_arnoldi_supports_real_linear_generators():
    generator = np.array([[-0.2, 0.4], [-0.1, -0.3]])
    state = np.array([1.0, -0.4])
    result = ArnoldiExponential(
        max_krylov_dimension=2, rtol=1.0e-13, atol=1.0e-15
    )(state, 0.0, 0.3, lambda values, time: generator @ values)

    np.testing.assert_allclose(result, expm(0.3 * generator) @ state, atol=1.0e-13)


@pytest.mark.parametrize("factory", [ShortIterativeLanczos, ArnoldiExponential])
def test_krylov_zero_state_and_zero_step(factory):
    integrator = factory()
    zero = np.zeros(3)
    result = integrator(zero, 0.0, 0.2, lambda values, time: values)
    np.testing.assert_array_equal(result, zero)
    assert integrator.last_operator_calls == 0
    assert integrator.last_accepted_substeps == 1

    state = np.arange(3.0)
    result = integrator(state, 0.0, 0.0, lambda values, time: values)
    np.testing.assert_array_equal(result, state)
    assert result is not state
    assert integrator.last_attempts == ()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ShortIterativeLanczos(max_krylov_dimension=0), "dimension"),
        (lambda: ArnoldiExponential(max_refinements=-1), "refinements"),
        (lambda: ShortIterativeLanczos(rtol=0.0, atol=0.0), "At least one"),
        (lambda: ArnoldiExponential(reorthogonalize=1), "boolean"),
        (lambda: ShortIterativeLanczos(inner_product=1), "inner_product"),
        (
            lambda: ArnoldiExponential(breakdown_tolerance=0.0),
            "breakdown_tolerance",
        ),
        (
            lambda: ShortIterativeLanczos(generator_factor=0.0),
            "generator_factor",
        ),
        (
            lambda: ShortIterativeLanczos(generator_factor=np.inf),
            "generator_factor",
        ),
        (
            lambda: ShortIterativeLanczos(generator_factor=True),
            "generator_factor",
        ),
    ],
)
def test_invalid_krylov_configuration(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


@pytest.mark.parametrize("factory", [ShortIterativeLanczos, ArnoldiExponential])
def test_krylov_rejects_shape_changing_right_hand_side(factory):
    with pytest.raises(ValueError, match="preserve"):
        factory()(np.ones(2), 0.0, 0.1, lambda values, time: np.ones(3))
