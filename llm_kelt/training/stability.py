"""Training stability detection and warnings.

Analyzes training metrics to detect instability patterns like gradient explosion,
loss spikes, training divergence, and overfitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from appinfra.log import Logger


# Overfitting thresholds
LOW_ENTROPY_THRESHOLD = 0.6  # Below this = model too deterministic
NEAR_ZERO_LOSS_THRESHOLD = 0.3  # Below this = likely memorizing
HIGH_ACCURACY_THRESHOLD = 0.95  # Above this = likely memorization
ENTROPY_DROP_RATE_THRESHOLD = 0.3  # Per epoch, above this = learning too aggressively


@dataclass
class StabilityReport:
    """Results of training stability analysis."""

    stable: bool
    warnings: list[str]
    nan_grad_norm_count: int = 0
    loss_spike_count: int = 0
    final_loss: float | None = None
    min_loss: float | None = None
    # Overfitting metrics
    final_entropy: float | None = None
    initial_entropy: float | None = None
    final_accuracy: float | None = None


def _is_nan(value: float | None) -> bool:
    """Check if value is NaN (handles None and string '.nan')."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in (".nan", "nan")
    return math.isnan(value)


def _get_float(log: dict, key: str) -> float | None:
    """Extract float value from log entry, handling NaN strings and invalid types."""
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
    # Guard against non-scalar types (dict, list) that would raise TypeError
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class EntropyEpochPair:
    """Aligned entropy and epoch values from the same log entry."""

    entropy: float
    epoch: float


@dataclass
class LogAnalysis:
    """Extracted metrics from training log history."""

    nan_grad_norm_count: int
    losses: list[float]
    entropies: list[float]
    accuracies: list[float]
    entropy_epoch_pairs: list[EntropyEpochPair]  # Aligned pairs for drop rate calc


def _valid_float(log: dict, key: str) -> float | None:
    """Get valid (non-NaN) float from log entry, or None."""
    value = _get_float(log, key)
    return value if value is not None and not _is_nan(value) else None


def _process_log_entry(
    log: dict,
    losses: list[float],
    entropies: list[float],
    accuracies: list[float],
    entropy_epoch_pairs: list[EntropyEpochPair],
) -> bool:
    """Process single log entry, appending valid values. Returns True if grad_norm is NaN."""
    if (loss := _valid_float(log, "loss")) is not None:
        losses.append(loss)
    if (entropy := _valid_float(log, "entropy")) is not None:
        entropies.append(entropy)
    if (accuracy := _valid_float(log, "mean_token_accuracy")) is not None:
        accuracies.append(accuracy)
    # Aligned pair: both entropy and epoch valid in same entry
    if entropy is not None and (epoch := _valid_float(log, "epoch")) is not None:
        entropy_epoch_pairs.append(EntropyEpochPair(entropy=entropy, epoch=epoch))
    # Check for NaN gradient norm
    grad_norm = _get_float(log, "grad_norm")
    return grad_norm is not None and _is_nan(grad_norm)


def _analyze_log_entries(log_history: list[dict]) -> LogAnalysis:
    """Extract metrics from log history for stability analysis (single pass)."""
    losses: list[float] = []
    entropies: list[float] = []
    accuracies: list[float] = []
    entropy_epoch_pairs: list[EntropyEpochPair] = []
    nan_count = sum(
        _process_log_entry(log, losses, entropies, accuracies, entropy_epoch_pairs)
        for log in log_history
    )
    return LogAnalysis(
        nan_grad_norm_count=nan_count,
        losses=losses,
        entropies=entropies,
        accuracies=accuracies,
        entropy_epoch_pairs=entropy_epoch_pairs,
    )


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


def _check_divergence(final_loss: float, min_loss: float) -> str | None:
    """Check for training divergence and return warning message if detected."""
    if min_loss == 0 and final_loss > 0:
        return (
            f"WARNING: Final loss ({final_loss:.2f}) rebounded from zero minimum. "
            "Training may have diverged."
        )
    if min_loss > 0:
        divergence_ratio = final_loss / min_loss
        if divergence_ratio > 3.0:
            return (
                f"WARNING: Final loss ({final_loss:.2f}) is {divergence_ratio:.1f}x higher "
                f"than minimum achieved ({min_loss:.2f}). Training may have diverged."
            )
    return None


def _check_low_entropy(entropies: list[float]) -> str | None:
    """Check for low entropy (model too deterministic)."""
    if not entropies:
        return None
    final = entropies[-1]
    if final < LOW_ENTROPY_THRESHOLD:
        return (
            f"WARNING: Low entropy ({final:.4f} < {LOW_ENTROPY_THRESHOLD}). "
            "Model may be overfitting - outputs will be repetitive/deterministic. "
            "Consider fewer epochs or lower learning rate."
        )
    return None


def _check_low_loss(losses: list[float]) -> str | None:
    """Check for near-zero loss (memorization)."""
    if not losses:
        return None
    final = losses[-1]
    if final < NEAR_ZERO_LOSS_THRESHOLD:
        return (
            f"WARNING: Very low loss ({final:.4f} < {NEAR_ZERO_LOSS_THRESHOLD}). "
            "Model may be memorizing training data instead of generalizing. "
            "Consider fewer epochs or more training data."
        )
    return None


def _check_high_accuracy(accuracies: list[float]) -> str | None:
    """Check for high accuracy (memorization indicator)."""
    if not accuracies:
        return None
    final = accuracies[-1]
    if final > HIGH_ACCURACY_THRESHOLD:
        return (
            f"WARNING: Very high token accuracy ({final:.1%} > {HIGH_ACCURACY_THRESHOLD:.0%}). "
            "Model may be memorizing training data. Consider fewer epochs."
        )
    return None


def _check_entropy_drop_rate(pairs: list[EntropyEpochPair]) -> str | None:
    """Check for rapid entropy drop (learning too aggressively).

    Uses aligned entropy/epoch pairs to ensure values come from the same log entries.
    Scans consecutive intervals to detect sharp mid-training drops that might recover.
    """
    if len(pairs) < 2:
        return None
    # Sort by epoch and find max per-interval drop rate
    ordered = sorted(pairs, key=lambda p: p.epoch)
    max_drop_rate = 0.0
    for prev, curr in zip(ordered, ordered[1:]):
        delta_epoch = curr.epoch - prev.epoch
        if delta_epoch <= 0:
            continue
        interval_drop = (prev.entropy - curr.entropy) / delta_epoch
        max_drop_rate = max(max_drop_rate, interval_drop)
    if max_drop_rate > ENTROPY_DROP_RATE_THRESHOLD:
        return (
            f"WARNING: Rapid entropy drop ({max_drop_rate:.4f}/epoch > "
            f"{ENTROPY_DROP_RATE_THRESHOLD}/epoch). Model learning too aggressively. "
            "Consider lower learning rate."
        )
    return None


def _check_overfit(analysis: LogAnalysis) -> list[str]:
    """Check for overfitting indicators and return warning messages."""
    checks = [
        _check_low_entropy(analysis.entropies),
        _check_low_loss(analysis.losses),
        _check_high_accuracy(analysis.accuracies),
        _check_entropy_drop_rate(analysis.entropy_epoch_pairs),
    ]
    return [w for w in checks if w is not None]


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

    if final_loss is not None and min_loss is not None:
        divergence_warning = _check_divergence(final_loss, min_loss)
        if divergence_warning:
            warnings.append(divergence_warning)

    return warnings


def check_training_stability(
    log_history: list[dict],
    loss_spike_threshold: float = 5.0,
    high_loss_threshold: float = 5.0,
) -> StabilityReport:
    """Analyze training log history for instability and overfitting patterns.

    Detects: NaN gradient norms, loss spikes, high final loss, divergence,
    low entropy, near-zero loss, high accuracy, rapid entropy drop.

    Args:
        log_history: List of training log entries from trainer.state.log_history.
        loss_spike_threshold: Multiplier to detect loss spikes (default 5x).
        high_loss_threshold: Absolute loss value considered high (default 5.0).

    Returns:
        StabilityReport with stability status and any warnings.
    """
    analysis = _analyze_log_entries(log_history)
    loss_spike_count = _count_loss_spikes(analysis.losses, loss_spike_threshold)

    final_loss = analysis.losses[-1] if analysis.losses else None
    min_loss = min(analysis.losses) if analysis.losses else None

    # Stability warnings (gradient explosion, divergence, etc.)
    warnings = _generate_warnings(
        analysis.nan_grad_norm_count,
        loss_spike_count,
        final_loss,
        min_loss,
        loss_spike_threshold,
        high_loss_threshold,
    )

    # Overfitting warnings (low entropy, memorization, etc.)
    warnings.extend(_check_overfit(analysis))

    return StabilityReport(
        stable=len(warnings) == 0,
        warnings=warnings,
        nan_grad_norm_count=analysis.nan_grad_norm_count,
        loss_spike_count=loss_spike_count,
        final_loss=final_loss,
        min_loss=min_loss,
        final_entropy=analysis.entropies[-1] if analysis.entropies else None,
        initial_entropy=analysis.entropies[0] if analysis.entropies else None,
        final_accuracy=analysis.accuracies[-1] if analysis.accuracies else None,
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
