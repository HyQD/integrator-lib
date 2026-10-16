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


class IMCMF1:
    r"""First-order constant-mean-field implicit-midpoint propagation.

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


class IMCMF2:
    r"""Second-order predictor-corrector constant-mean-field propagation.

    One predictor half-step advances from ``t`` to ``t + step_size/2`` with
    the operator frozen at the initial state. The predicted midpoint state is
    then frozen, and the corrector restarts from the initial state and covers
    the complete interval with one whole step or two half-steps.

    ``corrector_steps`` selects one whole corrector step or two corrector
    half-steps. Both choices use the same operator frozen at the predicted
    midpoint state.

    Parameters
    ----------
    predictor_integrator, corrector_integrator : callable
        Step callables with signature ``step(state, time, step_size, rhs)``.
        Separate objects allow their preconditioners to match the predictor
        and corrector substep sizes.
    corrector_steps : {1, 2}, optional
        Number of implicit-midpoint steps used to cover the complete corrector
        interval. Defaults to two.
    """

    def __init__(
        self,
        predictor_integrator,
        corrector_integrator,
        *,
        corrector_steps=2,
    ):
        if not callable(predictor_integrator):
            raise TypeError("predictor_integrator must be callable.")
        if not callable(corrector_integrator):
            raise TypeError("corrector_integrator must be callable.")
        if isinstance(corrector_steps, (bool, np.bool_)) or not isinstance(
            corrector_steps, (Integral, np.integer)
        ):
            raise TypeError("corrector_steps must be 1 or 2.")
        corrector_steps = int(corrector_steps)
        if corrector_steps not in {1, 2}:
            raise ValueError("corrector_steps must be 1 or 2.")
        self.predictor_integrator = predictor_integrator
        self.corrector_integrator = corrector_integrator
        self.corrector_steps = corrector_steps
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
        predictor = _step(
            self.predictor_integrator,
            values,
            time,
            step_size / 2,
            initial_rhs,
        )
        predictor_diagnostics = _linear_diagnostics(self.predictor_integrator)

        midpoint_rhs = _frozen_rhs(freeze, predictor)
        corrector = values
        corrector_step = step_size / self.corrector_steps
        corrector_iterations = []
        corrector_residuals = []
        for substep in range(self.corrector_steps):
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

        self.last_predictor_iterations = (predictor_diagnostics[0],)
        self.last_predictor_residual_norms = (predictor_diagnostics[1],)
        self.last_corrector_iterations = tuple(corrector_iterations)
        self.last_corrector_residual_norms = tuple(corrector_residuals)
        return corrector
