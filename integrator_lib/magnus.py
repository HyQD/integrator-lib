"""Commutator-free Magnus integration for linear time-dependent systems."""

from collections.abc import Callable

import numpy as np

from ._validation import finite_real, matching_array, state_array


_SQRT_THREE = float(np.sqrt(3.0))
_GAUSS_NODES = (
    0.5 - _SQRT_THREE / 6.0,
    0.5 + _SQRT_THREE / 6.0,
)
_LOW_WEIGHT = (3.0 - 2.0 * _SQRT_THREE) / 12.0
_HIGH_WEIGHT = (3.0 + 2.0 * _SQRT_THREE) / 12.0
_STAGE_WEIGHTS = (
    (_HIGH_WEIGHT, _LOW_WEIGHT),
    (_LOW_WEIGHT, _HIGH_WEIGHT),
)


class CommutatorFreeMagnus4:
    r"""Fourth-order commutator-free Magnus integration.

    For a linear nonautonomous equation

    .. math::

        y'(t) = L(t)y(t),

    one step evaluates ``L`` at the two fourth-order Gauss--Legendre nodes
    and applies two exponentials of real linear combinations of those
    operators.  No operator commutator or matrix representation is required.

    Parameters
    ----------
    exponential_integrator : callable
        Exponential stepper with signature
        ``step(state, time, step_size, linear_rhs)``.  The supplied
        ``linear_rhs`` is frozen during each exponential stage.  Examples are
        :class:`ShortIterativeLanczos` for Hermitian Hamiltonians and
        :class:`ArnoldiExponential` for general generators.
    linear_combination_factory : callable, optional
        Optimization hook with signature
        ``factory(linear_rhs, nodes, weights)``.  It must return a callable
        ``action(state)`` representing
        ``sum(weight * L(node) @ state)``.  The default evaluates
        ``linear_rhs`` once at each node for every action.  Applications with
        a structured operator, such as ``L(t) = L0 + f(t) L1``, may provide a
        factory that combines scalar coefficients before applying the
        expensive time-independent actions.

    Notes
    -----
    The first applied exponential is dominated by the earlier Gauss node and
    the second by the later node.  Reversing this application order reduces
    the method to second order for noncommuting time-dependent operators.

    Krylov diagnostics exposed by the configured exponential stepper are
    captured after both stages.  Aggregate attributes named
    ``last_operator_calls``, ``last_accepted_substeps``,
    ``last_rejected_substeps``, ``last_refinement_depth``,
    ``last_krylov_dimension``, and ``last_error_estimate`` follow the existing
    Krylov exponential convention when the wrapped stepper provides them.
    Per-stage values are available through the corresponding
    ``last_stage_*`` tuples.
    """

    gauss_nodes = _GAUSS_NODES
    stage_weights = _STAGE_WEIGHTS

    def __init__(
        self,
        exponential_integrator,
        *,
        linear_combination_factory=None,
    ):
        if not callable(exponential_integrator):
            raise TypeError("exponential_integrator must be callable.")
        if linear_combination_factory is not None and not callable(
            linear_combination_factory
        ):
            raise TypeError("linear_combination_factory must be callable or None.")
        self.exponential_integrator = exponential_integrator
        self.linear_combination_factory = (
            _default_linear_combination_factory
            if linear_combination_factory is None
            else linear_combination_factory
        )
        self._reset_diagnostics()

    def __call__(
        self,
        state,
        time,
        step_size,
        linear_rhs: Callable,
    ):
        """Advance one CF4 step without modifying *state*."""

        if not callable(linear_rhs):
            raise TypeError("linear_rhs must be callable.")
        values = state_array(state).copy()
        shape = values.shape
        time = finite_real(time, "time")
        step_size = finite_real(step_size, "step_size")
        self._reset_diagnostics()

        if step_size == 0.0:
            return values

        nodes = tuple(time + node * step_size for node in self.gauss_nodes)
        try:
            for weights in self.stage_weights:
                action = self.linear_combination_factory(
                    linear_rhs,
                    nodes,
                    weights,
                )
                if not callable(action):
                    raise TypeError(
                        "linear_combination_factory must return a callable."
                    )

                def frozen_rhs(argument, evaluation_time, action=action):
                    del evaluation_time
                    return action(argument)

                try:
                    values = self.exponential_integrator(
                        values,
                        time,
                        step_size,
                        frozen_rhs,
                    )
                finally:
                    self._capture_stage_diagnostics()
                values = matching_array(
                    values,
                    shape,
                    "exponential_integrator",
                )
        finally:
            self._aggregate_diagnostics()
        return values.copy()

    def _reset_diagnostics(self):
        self.last_stage_attempts = ()
        self.last_stage_operator_calls = ()
        self.last_stage_accepted_substeps = ()
        self.last_stage_rejected_substeps = ()
        self.last_stage_refinement_depths = ()
        self.last_stage_krylov_dimensions = ()
        self.last_stage_error_estimates = ()
        self.last_attempts = ()
        self.last_operator_calls = 0
        self.last_accepted_substeps = 0
        self.last_rejected_substeps = 0
        self.last_refinement_depth = 0
        self.last_krylov_dimension = 0
        self.last_error_estimate = 0.0

    def _capture_stage_diagnostics(self):
        stepper = self.exponential_integrator
        self.last_stage_attempts += (getattr(stepper, "last_attempts", None),)
        self.last_stage_operator_calls += (
            getattr(stepper, "last_operator_calls", None),
        )
        self.last_stage_accepted_substeps += (
            getattr(stepper, "last_accepted_substeps", None),
        )
        self.last_stage_rejected_substeps += (
            getattr(stepper, "last_rejected_substeps", None),
        )
        self.last_stage_refinement_depths += (
            getattr(stepper, "last_refinement_depth", None),
        )
        self.last_stage_krylov_dimensions += (
            getattr(stepper, "last_krylov_dimension", None),
        )
        self.last_stage_error_estimates += (
            getattr(stepper, "last_error_estimate", None),
        )

    def _aggregate_diagnostics(self):
        self.last_attempts = _flatten(self.last_stage_attempts)
        self.last_operator_calls = _sum_or_none(self.last_stage_operator_calls)
        self.last_accepted_substeps = _sum_or_none(
            self.last_stage_accepted_substeps
        )
        self.last_rejected_substeps = _sum_or_none(
            self.last_stage_rejected_substeps
        )
        self.last_refinement_depth = _max_or_none(
            self.last_stage_refinement_depths
        )
        self.last_krylov_dimension = _max_or_none(
            self.last_stage_krylov_dimensions
        )
        self.last_error_estimate = _sum_or_none(
            self.last_stage_error_estimates
        )


def _default_linear_combination_factory(linear_rhs, nodes, weights):
    def action(state):
        shape = np.asarray(state).shape
        result = None
        for node, weight in zip(nodes, weights, strict=True):
            term = matching_array(linear_rhs(state, node), shape, "linear_rhs")
            weighted = weight * term
            result = weighted if result is None else result + weighted
        return result

    return action


def _sum_or_none(values):
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _max_or_none(values):
    if not values or any(value is None for value in values):
        return None
    return max(values)


def _flatten(values):
    if not values or any(value is None for value in values):
        return None
    return tuple(item for stage in values for item in stage)
