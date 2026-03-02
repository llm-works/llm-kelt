"""Training stability detection and warnings.

Analyzes training metrics to detect instability patterns like gradient explosion,
loss spikes, and training divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from appinfra.log import Logger


@dataclass
class StabilityReport:
    """Results of training stability analysis."""

    stable: bool
    warnings: list[str]
    nan_grad_norm_count: int = 0
    loss_spike_count: int = 0
    final_loss: float | None = None
    min_loss: float | None = None


def _is_nan(value: float | None) -> bool:
    """Check if value is NaN (handles None and string '.nan')."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in (".nan", "nan")
    return math.isnan(value)


def _get_float(log: dict, key: str) -> float | None:
    """Extract float value from log entry, handling NaN strings."""
    value = log.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in (".nan", "nan"):
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return None
    return float(value)


def _analyze_log_entries(log_history: list[dict]) -> tuple[int, list[float]]:
    """Extract NaN grad_norm count and valid losses from log history."""
    nan_grad_norm_count = 0
    losses: list[float] = []

    for log in log_history:
        grad_norm = _get_float(log, "grad_norm")
        if grad_norm is not None and _is_nan(grad_norm):
            nan_grad_norm_count += 1

        loss = _get_float(log, "loss")
        if loss is not None and not _is_nan(loss):
            losses.append(loss)

    return nan_grad_norm_count, losses


def _count_loss_spikes(losses: list[float], threshold: float) -> int:
    """Count loss spikes (compared to rolling average of previous 5 steps)."""
    if len(losses) <= 5:
        return 0

    spike_count = 0
    for i in range(5, len(losses)):
        window_avg = sum(losses[i - 5 : i]) / 5
        if window_avg > 0 and losses[i] / window_avg > threshold:
            spike_count += 1
    return spike_count


def _generate_warnings(
    nan_count: int,
    spike_count: int,
    final_loss: float | None,
    min_loss: float | None,
    loss_spike_threshold: float,
    high_loss_threshold: float,
) -> list[str]:
    """Generate warning messages based on detected instability patterns."""
    warnings: list[str] = []

    if nan_count > 0:
        warnings.append(
            f"CRITICAL: Detected {nan_count} steps with NaN gradient norm. "
            "Gradients exploded - model weights are likely corrupted. "
            "Consider lower learning rate or enable gradient clipping (max_grad_norm)."
        )

    if spike_count > 0:
        warnings.append(
            f"WARNING: Detected {spike_count} loss spikes (>{loss_spike_threshold}x increase). "
            "Training was unstable. Consider lower learning rate."
        )

    if final_loss is not None and final_loss > high_loss_threshold:
        warnings.append(
            f"WARNING: Final loss ({final_loss:.2f}) is high. "
            "Model may not have converged properly."
        )

    if final_loss is not None and min_loss is not None and min_loss > 0:
        divergence_ratio = final_loss / min_loss
        if divergence_ratio > 3.0:
            warnings.append(
                f"WARNING: Final loss ({final_loss:.2f}) is {divergence_ratio:.1f}x higher "
                f"than minimum achieved ({min_loss:.2f}). Training may have diverged."
            )

    return warnings


def check_training_stability(
    log_history: list[dict],
    loss_spike_threshold: float = 5.0,
    high_loss_threshold: float = 5.0,
) -> StabilityReport:
    """Analyze training log history for instability patterns.

    Detects: NaN gradient norms, loss spikes, high final loss, divergence.

    Args:
        log_history: List of training log entries from trainer.state.log_history.
        loss_spike_threshold: Multiplier to detect loss spikes (default 5x).
        high_loss_threshold: Absolute loss value considered high (default 5.0).

    Returns:
        StabilityReport with stability status and any warnings.
    """
    nan_grad_norm_count, losses = _analyze_log_entries(log_history)
    loss_spike_count = _count_loss_spikes(losses, loss_spike_threshold)

    final_loss = losses[-1] if losses else None
    min_loss = min(losses) if losses else None

    warnings = _generate_warnings(
        nan_grad_norm_count,
        loss_spike_count,
        final_loss,
        min_loss,
        loss_spike_threshold,
        high_loss_threshold,
    )

    return StabilityReport(
        stable=len(warnings) == 0,
        warnings=warnings,
        nan_grad_norm_count=nan_grad_norm_count,
        loss_spike_count=loss_spike_count,
        final_loss=final_loss,
        min_loss=min_loss,
    )


def log_stability_warnings(lg: Logger, report: StabilityReport) -> None:
    """Log stability warnings at appropriate levels.

    Args:
        lg: Logger instance.
        report: StabilityReport from check_training_stability.
    """
    if report.stable:
        return

    for warning in report.warnings:
        if warning.startswith("CRITICAL"):
            lg.error(warning)
        else:
            lg.warning(warning)

    if report.nan_grad_norm_count > 0:
        lg.error(
            "training_instability_detected",
            extra={
                "nan_grad_norm_count": report.nan_grad_norm_count,
                "loss_spike_count": report.loss_spike_count,
                "final_loss": report.final_loss,
                "min_loss": report.min_loss,
                "recommendation": "Retrain with lower learning rate and gradient clipping enabled",
            },
        )
