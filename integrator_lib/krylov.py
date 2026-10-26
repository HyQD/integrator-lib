"""Matrix-free exponential propagation in Krylov subspaces."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ._validation import finite_real, matching_array, positive_real, state_array


class KrylovConvergenceError(RuntimeError):
    """Raised when Krylov propagation fails after all step refinements."""


@dataclass(frozen=True)
class KrylovAttempt:
    """Diagnostics for one attempted exponential substep."""

    time: float
    step_size: float
    refinement_depth: int
    krylov_dimension: int
    error_estimate: float
    tolerance: float
    operator_calls: int
    converged: bool
    breakdown: bool


class _KrylovExponential:
    """Shared adaptive-substepping machinery for Krylov propagators."""

    def __init__(
        self,
        *,
        max_krylov_dimension=20,
        rtol=1.0e-10,
        atol=0.0,
        max_refinements=8,
        reorthogonalize=True,
        inner_product=None,
        breakdown_tolerance=1.0e-14,
    ):
        self.max_krylov_dimension = _integer(
            max_krylov_dimension, "max_krylov_dimension", minimum=1
        )
        self.rtol = positive_real(rtol, "rtol", allow_zero=True)
        self.atol = positive_real(atol, "atol", allow_zero=True)
        if self.rtol == self.atol == 0.0:
            raise ValueError("At least one of rtol and atol must be positive.")
        self.max_refinements = _integer(
            max_refinements, "max_refinements", minimum=0
        )
        if not isinstance(reorthogonalize, (bool, np.bool_)):
            raise TypeError("reorthogonalize must be boolean.")
        if inner_product is not None and not callable(inner_product):
            raise TypeError("inner_product must be callable or None.")
        self.reorthogonalize = bool(reorthogonalize)
        self.inner_product = np.vdot if inner_product is None else inner_product
        self.breakdown_tolerance = positive_real(
            breakdown_tolerance, "breakdown_tolerance"
        )
        self.last_attempts = ()
        self.last_operator_calls = None
        self.last_accepted_substeps = None
        self.last_rejected_substeps = None
        self.last_refinement_depth = None
        self.last_krylov_dimension = None
        self.last_error_estimate = None

    def __call__(self, state, time, step_size, linear_rhs: Callable):
        """Advance one interval without modifying *state*."""

        if not callable(linear_rhs):
            raise TypeError("linear_rhs must be callable.")
        values = state_array(state)
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        attempts = []

        if step_size == 0.0:
            self._store_diagnostics(attempts)
            return values.copy()

        try:
            result = self._advance(
                values,
                time,
                step_size,
                linear_rhs,
                refinement_depth=0,
                tolerance_scale=1.0,
                attempts=attempts,
            )
        except Exception:
            self._store_diagnostics(attempts)
            raise
        self._store_diagnostics(attempts)
        return result

    def _advance(
        self,
        state,
        time,
        step_size,
        linear_rhs,
        *,
        refinement_depth,
        tolerance_scale,
        attempts,
    ):
        result, attempt = self._krylov_step(
            state,
            time,
            step_size,
            linear_rhs,
            refinement_depth,
            tolerance_scale,
        )
        attempts.append(attempt)
        if attempt.converged:
            return result
        if refinement_depth >= self.max_refinements:
            raise KrylovConvergenceError(
                "Krylov exponential did not converge with "
                f"max_krylov_dimension={self.max_krylov_dimension} after "
                f"{self.max_refinements} step refinements; final estimated "
                f"error={attempt.error_estimate:.3e}, "
                f"tolerance={attempt.tolerance:.3e}."
            )

        half_step = 0.5 * step_size
        midpoint_state = self._advance(
            state,
            time,
            half_step,
            linear_rhs,
            refinement_depth=refinement_depth + 1,
            tolerance_scale=0.5 * tolerance_scale,
            attempts=attempts,
        )
        return self._advance(
            midpoint_state,
            time + half_step,
            half_step,
            linear_rhs,
            refinement_depth=refinement_depth + 1,
            tolerance_scale=0.5 * tolerance_scale,
            attempts=attempts,
        )

    def _norm(self, value):
        inner = self.inner_product(value, value)
        array = np.asarray(inner)
        if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
            raise TypeError("inner_product must return a numeric scalar.")
        inner = complex(array)
        if not np.isfinite(inner):
            raise ValueError("inner_product must return a finite scalar.")
        scale = max(1.0, abs(inner.real))
        if abs(inner.imag) > 1.0e-11 * scale:
            raise ValueError(
                "inner_product(state, state) must be real to numerical precision."
            )
        if inner.real < -1.0e-13 * scale:
            raise ValueError("inner_product must define a positive norm.")
        return float(np.sqrt(max(0.0, inner.real)))

    def _inner(self, left, right):
        value = np.asarray(self.inner_product(left, right))
        if value.ndim != 0 or not np.issubdtype(value.dtype, np.number):
            raise TypeError("inner_product must return a numeric scalar.")
        value = complex(value)
        if not np.isfinite(value):
            raise ValueError("inner_product must return a finite scalar.")
        return value

    def _linear_combination(self, basis, coefficients, shape):
        dtype = np.result_type(*(vector.dtype for vector in basis), coefficients.dtype)
        result = np.zeros(shape, dtype=dtype)
        for coefficient, vector in zip(coefficients, basis, strict=True):
            result += coefficient * vector
        return result

    def _convergence(
        self, approximation, error_estimate, initial_norm, tolerance_scale
    ):
        approximation_norm = self._norm(approximation)
        tolerance = tolerance_scale * (
            self.atol + self.rtol * max(initial_norm, approximation_norm)
        )
        error_estimate = float(error_estimate)
        return error_estimate, tolerance, error_estimate <= tolerance

    def _store_diagnostics(self, attempts):
        self.last_attempts = tuple(attempts)
        self.last_operator_calls = sum(item.operator_calls for item in attempts)
        accepted = [item for item in attempts if item.converged]
        self.last_accepted_substeps = len(accepted)
        self.last_rejected_substeps = len(attempts) - len(accepted)
        self.last_refinement_depth = max(
            (item.refinement_depth for item in attempts), default=0
        )
        if accepted:
            self.last_krylov_dimension = max(
                item.krylov_dimension for item in accepted
            )
            self.last_error_estimate = sum(
                item.error_estimate for item in accepted
            )
        else:
            self.last_krylov_dimension = 0
            self.last_error_estimate = None


class ShortIterativeLanczos(_KrylovExponential):
    r"""Short-iterative Lanczos propagation for Hermitian TDSE Hamiltonians.

    ``linear_rhs(state, time)`` must represent the anti-Hermitian TDSE
    generator ``-1j * H(time) @ state``, where ``H`` is Hermitian under the
    configured inner product. The Hamiltonian is sampled at the midpoint of
    each accepted or refined substep.

    The Krylov approximation is increased up to ``max_krylov_dimension``.
    Its convergence estimate is formed from the endpoint Krylov residual. If
    it does not meet ``atol + rtol*norm``, the time interval is bisected
    recursively up to ``max_refinements`` times.
    """

    def _krylov_step(
        self,
        state,
        time,
        step_size,
        linear_rhs,
        refinement_depth,
        tolerance_scale,
    ):
        shape = state.shape
        initial_norm = self._norm(state)
        midpoint = time + 0.5 * step_size
        if initial_norm == 0.0:
            attempt = KrylovAttempt(
                time,
                step_size,
                refinement_depth,
                0,
                0.0,
                self.atol,
                0,
                True,
                True,
            )
            return state.copy(), attempt

        operator_calls = 0

        def hamiltonian_action(argument):
            nonlocal operator_calls
            operator_calls += 1
            derivative = matching_array(
                linear_rhs(argument, midpoint), shape, "linear_rhs"
            )
            return 1j * derivative

        basis = [state / initial_norm]
        alphas = []
        betas = []
        approximation = None
        error = np.inf
        tolerance = self.atol + self.rtol * initial_norm
        breakdown = False

        for index in range(self.max_krylov_dimension):
            vector = basis[index]
            residual = hamiltonian_action(vector)
            if index:
                residual = residual - betas[index - 1] * basis[index - 1]
            alpha = self._inner(vector, residual)
            alpha_scale = max(1.0, abs(alpha.real))
            if abs(alpha.imag) > 1.0e-10 * alpha_scale:
                raise ValueError(
                    "ShortIterativeLanczos requires a Hermitian Hamiltonian "
                    "under the configured inner product."
                )
            alpha = float(alpha.real)
            alphas.append(alpha)
            residual = residual - alpha * vector

            if self.reorthogonalize:
                for basis_vector in basis:
                    residual = residual - (
                        self._inner(basis_vector, residual) * basis_vector
                    )

            beta = self._norm(residual)
            dimension = index + 1
            projected = np.diag(np.asarray(alphas))
            if betas:
                off_diagonal = np.asarray(betas[: dimension - 1])
                projected += np.diag(off_diagonal, 1)
                projected += np.diag(off_diagonal, -1)
            coefficients = initial_norm * expm(
                -1j * step_size * projected
            )[:, 0]
            approximation = self._linear_combination(
                basis, coefficients, shape
            )
            residual_estimate = (
                abs(step_size) * beta * abs(coefficients[-1])
            )
            error, tolerance, converged = self._convergence(
                approximation,
                residual_estimate,
                initial_norm,
                tolerance_scale,
            )

            breakdown = beta <= self.breakdown_tolerance
            if breakdown:
                error = 0.0
                converged = True
            if converged:
                break
            if dimension == self.max_krylov_dimension:
                break
            betas.append(beta)
            basis.append(residual / beta)

        attempt = KrylovAttempt(
            time,
            step_size,
            refinement_depth,
            dimension,
            float(error),
            float(tolerance),
            operator_calls,
            bool(converged),
            breakdown,
        )
        return approximation, attempt


class ArnoldiExponential(_KrylovExponential):
    r"""Arnoldi exponential propagation for a general linear generator.

    ``linear_rhs(state, time)`` represents the possibly non-Hermitian
    generator ``L(time) @ state`` in ``dstate/dt = L(t) state``. This includes
    TDSE Hamiltonians containing a complex absorbing potential. The generator
    is sampled at the midpoint of each accepted or refined substep.
    """

    def _krylov_step(
        self,
        state,
        time,
        step_size,
        linear_rhs,
        refinement_depth,
        tolerance_scale,
    ):
        shape = state.shape
        initial_norm = self._norm(state)
        midpoint = time + 0.5 * step_size
        if initial_norm == 0.0:
            attempt = KrylovAttempt(
                time,
                step_size,
                refinement_depth,
                0,
                0.0,
                self.atol,
                0,
                True,
                True,
            )
            return state.copy(), attempt

        operator_calls = 0

        def generator_action(argument):
            nonlocal operator_calls
            operator_calls += 1
            return matching_array(
                linear_rhs(argument, midpoint), shape, "linear_rhs"
            )

        basis = [state / initial_norm]
        hessenberg = np.zeros(
            (self.max_krylov_dimension + 1, self.max_krylov_dimension),
            dtype=complex,
        )
        approximation = None
        error = np.inf
        tolerance = self.atol + self.rtol * initial_norm
        breakdown = False

        for index in range(self.max_krylov_dimension):
            residual = generator_action(basis[index])
            passes = 2 if self.reorthogonalize else 1
            for _ in range(passes):
                for row, basis_vector in enumerate(basis):
                    coefficient = self._inner(basis_vector, residual)
                    hessenberg[row, index] += coefficient
                    residual = residual - coefficient * basis_vector

            beta = self._norm(residual)
            hessenberg[index + 1, index] = beta
            dimension = index + 1
            projected = hessenberg[:dimension, :dimension]
            coefficients = initial_norm * expm(step_size * projected)[:, 0]
            approximation = self._linear_combination(
                basis, coefficients, shape
            )
            residual_estimate = (
                abs(step_size) * beta * abs(coefficients[-1])
            )
            error, tolerance, converged = self._convergence(
                approximation,
                residual_estimate,
                initial_norm,
                tolerance_scale,
            )

            breakdown = beta <= self.breakdown_tolerance
            if breakdown:
                error = 0.0
                converged = True
            if converged:
                break
            if dimension == self.max_krylov_dimension:
                break
            basis.append(residual / beta)

        attempt = KrylovAttempt(
            time,
            step_size,
            refinement_depth,
            dimension,
            float(error),
            float(tolerance),
            operator_calls,
            bool(converged),
            breakdown,
        )
        return approximation, attempt


def _integer(value, name, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value
