"""Crank--Nicolson propagation with frozen state-dependent operators."""

from collections.abc import Callable
from numbers import Integral

import numpy as np

from ._validation import finite_real, matching_array, state_array


def _step(stepper, state, time, step_size, rhs):
    result = stepper(state, time, step_size, rhs)
    return matching_array(result, state.shape, "integrator")


def _linear_diagnostics(stepper):
    return (
        getattr(stepper, "last_iterations", None),
        getattr(stepper, "last_residual_norm", None),
    )


def _frozen_rhs(freeze, reference_state):
    rhs = freeze(reference_state)
    if not callable(rhs):
        raise TypeError("freeze must return a callable linear right-hand side.")
    return rhs


class CNCMF1:
    r"""First-order constant-mean-field Crank--Nicolson propagation.

    ``freeze(reference_state)`` must return a linear right-hand side with
    signature ``rhs(trial_state, time)``. The returned operator is frozen at
    the state supplied to ``freeze`` but may retain explicit time dependence.

    One outer step freezes the operator at the initial state and advances one
    Crank--Nicolson step over the complete interval.

    Parameters
    ----------
    crank_nicolson : callable
        Step callable with signature ``step(state, time, step_size, rhs)``.
    """

    def __init__(self, crank_nicolson):
        if not callable(crank_nicolson):
            raise TypeError("crank_nicolson must be callable.")
        self.crank_nicolson = crank_nicolson
        self.last_iterations = None
        self.last_residual_norm = None

    def __call__(self, state, time, step_size, freeze: Callable):
        """Advance one step with the operator frozen at the initial state."""
        if not callable(freeze):
            raise TypeError("freeze must be callable.")
        values = state_array(state).copy()
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        rhs = _frozen_rhs(freeze, values)
        result = _step(self.crank_nicolson, values, time, step_size, rhs)
        self.last_iterations, self.last_residual_norm = _linear_diagnostics(
            self.crank_nicolson
        )
        return result


class CNCMF2:
    r"""Second-order predictor-corrector constant-mean-field propagation.

    A predictor advances from ``t`` to ``t + step_size/2`` with the operator
    frozen at the initial state. The predicted midpoint state is then frozen,
    and the corrector restarts from the initial state and takes two
    Crank--Nicolson half-steps over the complete interval.

    ``predictor_steps`` selects either one predictor half-step or two predictor
    quarter-steps. Both use the same operator frozen at the initial state.

    Parameters
    ----------
    predictor_integrator, corrector_integrator : callable
        Step callables with signature ``step(state, time, step_size, rhs)``.
        Separate objects allow their preconditioners to match the predictor
        and corrector substep sizes.
    predictor_steps : {1, 2}, optional
        Number of Crank--Nicolson steps used to reach the predicted midpoint.
        Defaults to two.
    """

    def __init__(
        self,
        predictor_integrator,
        corrector_integrator,
        *,
        predictor_steps=2,
    ):
        if not callable(predictor_integrator):
            raise TypeError("predictor_integrator must be callable.")
        if not callable(corrector_integrator):
            raise TypeError("corrector_integrator must be callable.")
        if isinstance(predictor_steps, (bool, np.bool_)) or not isinstance(
            predictor_steps, (Integral, np.integer)
        ):
            raise TypeError("predictor_steps must be 1 or 2.")
        predictor_steps = int(predictor_steps)
        if predictor_steps not in {1, 2}:
            raise ValueError("predictor_steps must be 1 or 2.")
        self.predictor_integrator = predictor_integrator
        self.corrector_integrator = corrector_integrator
        self.predictor_steps = predictor_steps
        self.last_predictor_iterations = ()
        self.last_predictor_residual_norms = ()
        self.last_corrector_iterations = ()
        self.last_corrector_residual_norms = ()

    def __call__(self, state, time, step_size, freeze: Callable):
        """Advance one second-order predictor-corrector step."""
        if not callable(freeze):
            raise TypeError("freeze must be callable.")
        values = state_array(state).copy()
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        self.last_predictor_iterations = ()
        self.last_predictor_residual_norms = ()
        self.last_corrector_iterations = ()
        self.last_corrector_residual_norms = ()

        initial_rhs = _frozen_rhs(freeze, values)
        predictor = values
        predictor_step = step_size / (2 * self.predictor_steps)
        predictor_iterations = []
        predictor_residuals = []
        for substep in range(self.predictor_steps):
            predictor = _step(
                self.predictor_integrator,
                predictor,
                time + substep * predictor_step,
                predictor_step,
                initial_rhs,
            )
            iterations, residual = _linear_diagnostics(self.predictor_integrator)
            predictor_iterations.append(iterations)
            predictor_residuals.append(residual)

        midpoint_rhs = _frozen_rhs(freeze, predictor)
        corrector = values
        corrector_step = step_size / 2
        corrector_iterations = []
        corrector_residuals = []
        for substep in range(2):
            corrector = _step(
                self.corrector_integrator,
                corrector,
                time + substep * corrector_step,
                corrector_step,
                midpoint_rhs,
            )
            iterations, residual = _linear_diagnostics(self.corrector_integrator)
            corrector_iterations.append(iterations)
            corrector_residuals.append(residual)

        self.last_predictor_iterations = tuple(predictor_iterations)
        self.last_predictor_residual_norms = tuple(predictor_residuals)
        self.last_corrector_iterations = tuple(corrector_iterations)
        self.last_corrector_residual_norms = tuple(corrector_residuals)
        return corrector
