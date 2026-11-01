"""First-order exponential Euler integration."""

from collections.abc import Callable

import numpy as np

from ._validation import finite_real, matching_array, state_array


class ExponentialEuler:
    r"""Advance a rate with a caller-selected matrix-free phi-function action.

    For a complete rate ``F(state, time)`` and a selected linear operator
    ``L(time)``, one step is

    .. math::

        y_{n+1} = y_n + h\,\varphi_1(hL)F(y_n, t_n).

    ``full_rhs`` supplies the complete rate ``F`` and is evaluated exactly
    once at the beginning of a nonzero step. ``linear_rhs`` supplies the
    action of ``L`` and is forwarded to ``phi1_action``. The configured action
    therefore controls operator sampling, Krylov tolerances, inner products,
    and Hermitian versus general-operator assumptions.

    Parameters
    ----------
    phi1_action : callable
        Callable with signature
        ``action(vector, time, step_size, linear_rhs)`` returning
        ``step_size * phi_1(step_size * L) @ vector``.

    Notes
    -----
    The class does not normalize the result or choose a splitting. For a
    semilinear equation ``y' = L y + N(y)``, ``full_rhs`` should return
    ``L y + N(y)``. A stationary-state application may instead provide its
    complete residual as ``full_rhs``.
    """

    def __init__(self, phi1_action):
        if not callable(phi1_action):
            raise TypeError("phi1_action must be callable.")
        self.phi1_action = phi1_action
        self._reset_diagnostics()

    def __call__(
        self,
        state,
        time,
        step_size,
        full_rhs: Callable,
        linear_rhs: Callable,
    ):
        """Advance one exponential-Euler step without modifying *state*."""

        if not callable(full_rhs):
            raise TypeError("full_rhs must be callable.")
        if not callable(linear_rhs):
            raise TypeError("linear_rhs must be callable.")
        values = state_array(state).copy()
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        self._reset_diagnostics()

        if step_size == 0.0:
            return values

        rate = matching_array(full_rhs(values, time), values.shape, "full_rhs")
        self.last_rhs_evaluations = 1
        try:
            correction = self.phi1_action(
                rate,
                time,
                step_size,
                linear_rhs,
            )
        finally:
            self._collect_phi1_diagnostics()
        correction = matching_array(
            correction,
            values.shape,
            "phi1_action",
        )
        return values + correction

    def _reset_diagnostics(self):
        self.last_rhs_evaluations = 0
        self.last_phi1_operator_calls = 0
        self.last_phi1_attempts = ()
        self.last_phi1_krylov_dimension = 0
        self.last_phi1_error_estimate = None

    def _collect_phi1_diagnostics(self):
        action = self.phi1_action
        self.last_phi1_operator_calls = getattr(
            action,
            "last_operator_calls",
            None,
        )
        self.last_phi1_attempts = getattr(action, "last_attempts", None)
        self.last_phi1_krylov_dimension = getattr(
            action,
            "last_krylov_dimension",
            None,
        )
        self.last_phi1_error_estimate = getattr(
            action,
            "last_error_estimate",
            None,
        )
