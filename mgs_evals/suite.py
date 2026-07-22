from __future__ import annotations

import inspect
from typing import Any

from .base import Metric


class EvalSuite:
    """Run a collection of metrics and aggregate results into a single dict.

    Each metric's compute() is called with only the kwargs it declares.
    Failed metrics are re-raised immediately unless raise_on_error=False,
    in which case errors are collected under results["errors"].
    """

    def __init__(self, metrics: list[Metric], raise_on_error: bool = True):
        self.metrics = metrics
        self.raise_on_error = raise_on_error

    def run(self, **data: Any) -> dict[str, Any]:
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for metric in self.metrics:
            filtered = _filter_for(metric.compute, data)
            try:
                out = metric.compute(**filtered)
                results.update(out)
            except Exception as exc:
                if self.raise_on_error:
                    raise
                errors[metric.name] = f"{type(exc).__name__}: {exc}"

        if errors:
            results["errors"] = errors
        return results

    def __repr__(self) -> str:
        names = [m.name for m in self.metrics]
        return f"EvalSuite([{', '.join(names)}])"


def _filter_for(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only the kwargs that fn's signature declares (excluding **_)"""
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_keyword:
        return kwargs
    valid = {
        name for name, p in params.items()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL,) and name != "self"
    }
    return {k: v for k, v in kwargs.items() if k in valid}