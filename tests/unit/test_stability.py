"""Tests for training stability detection."""

from llm_kelt.training.stability import check_training_stability


class TestCheckTrainingStability:
    """Tests for check_training_stability function."""

    def test_stable_training(self):
        """Stable training with decreasing loss."""
        log_history = [
            {"loss": 3.0, "grad_norm": 1.5, "step": 10},
            {"loss": 2.5, "grad_norm": 1.3, "step": 20},
            {"loss": 2.0, "grad_norm": 1.1, "step": 30},
            {"loss": 1.5, "grad_norm": 0.9, "step": 40},
            {"loss": 1.2, "grad_norm": 0.8, "step": 50},
            {"loss": 1.0, "grad_norm": 0.7, "step": 60},
        ]
        report = check_training_stability(log_history)
        assert report.stable is True
        assert report.warnings == []
        assert report.nan_grad_norm_count == 0
        assert report.loss_spike_count == 0

    def test_nan_grad_norm_detected(self):
        """Detects NaN gradient norms (gradient explosion)."""
        log_history = [
            {"loss": 1.0, "grad_norm": 1.5, "step": 10},
            {"loss": 1.0, "grad_norm": 1.3, "step": 20},
            {"loss": 5.0, "grad_norm": float("nan"), "step": 30},
            {"loss": 10.0, "grad_norm": float("nan"), "step": 40},
            {"loss": 20.0, "grad_norm": float("nan"), "step": 50},
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.nan_grad_norm_count == 3
        assert any("NaN gradient norm" in w for w in report.warnings)
        assert any("CRITICAL" in w for w in report.warnings)

    def test_nan_grad_norm_string_format(self):
        """Handles .nan string format from YAML."""
        log_history = [
            {"loss": 1.0, "grad_norm": 1.5, "step": 10},
            {"loss": 5.0, "grad_norm": ".nan", "step": 20},
            {"loss": 10.0, "grad_norm": "nan", "step": 30},
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.nan_grad_norm_count == 2

    def test_loss_spike_detected(self):
        """Detects sudden loss spikes."""
        log_history = [
            {"loss": 1.0, "step": 10},
            {"loss": 0.9, "step": 20},
            {"loss": 0.85, "step": 30},
            {"loss": 0.8, "step": 40},
            {"loss": 0.75, "step": 50},
            {"loss": 10.0, "step": 60},  # Spike: 10x increase
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.loss_spike_count >= 1
        assert any("loss spike" in w.lower() for w in report.warnings)

    def test_high_final_loss_warning(self):
        """Warns when final loss is high."""
        log_history = [
            {"loss": 3.0, "step": 10},
            {"loss": 4.0, "step": 20},
            {"loss": 5.0, "step": 30},
            {"loss": 6.0, "step": 40},
            {"loss": 7.0, "step": 50},
            {"loss": 8.0, "step": 60},  # High final loss
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.final_loss == 8.0
        assert any("Final loss" in w and "high" in w.lower() for w in report.warnings)

    def test_divergence_detected(self):
        """Detects when training diverges (final loss >> min loss)."""
        log_history = [
            {"loss": 2.0, "step": 10},
            {"loss": 1.5, "step": 20},
            {"loss": 1.0, "step": 30},
            {"loss": 0.8, "step": 40},
            {"loss": 0.5, "step": 50},  # Minimum
            {"loss": 2.0, "step": 60},  # Diverging: 4x min
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.min_loss == 0.5
        assert report.final_loss == 2.0
        assert any("higher than minimum" in w for w in report.warnings)

    def test_divergence_from_zero_min(self):
        """Detects divergence when minimum loss is zero and final rebounds."""
        log_history = [
            {"loss": 1.0, "step": 10},
            {"loss": 0.5, "step": 20},
            {"loss": 0.0, "step": 30},  # Zero minimum
            {"loss": 0.8, "step": 40},  # Rebounds
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.min_loss == 0.0
        assert report.final_loss == 0.8
        assert any("rebounded from zero" in w for w in report.warnings)

    def test_empty_history(self):
        """Handles empty log history."""
        report = check_training_stability([])
        assert report.stable is True
        assert report.warnings == []
        assert report.final_loss is None
        assert report.min_loss is None

    def test_missing_grad_norm(self):
        """Handles logs without grad_norm field."""
        log_history = [
            {"loss": 1.0, "step": 10},
            {"loss": 0.8, "step": 20},
            {"loss": 0.6, "step": 30},
        ]
        report = check_training_stability(log_history)
        assert report.stable is True
        assert report.nan_grad_norm_count == 0

    def test_realistic_explosion_scenario(self):
        """Simulates the actual failure mode from the jokester-p-sft training."""
        # Recreate the pattern: stable -> NaN grad_norm -> loss explosion
        log_history = [
            {"loss": 1.0, "grad_norm": 1.2, "step": 10},
            {"loss": 0.9, "grad_norm": 1.1, "step": 20},
            {"loss": 0.87, "grad_norm": 1.0, "step": 30},
            {"loss": 0.86, "grad_norm": 0.9, "step": 40},
            {"loss": 0.865, "grad_norm": ".nan", "step": 50},  # NaN starts
            {"loss": 4.34, "grad_norm": ".nan", "step": 60},
            {"loss": 16.88, "grad_norm": ".nan", "step": 70},
            {"loss": 41.6, "grad_norm": ".nan", "step": 80},
            {"loss": 20.0, "grad_norm": ".nan", "step": 90},
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert report.nan_grad_norm_count == 5
        assert report.loss_spike_count >= 1
        # Should have multiple warnings
        assert len(report.warnings) >= 2
        # Critical warning for NaN
        assert any("CRITICAL" in w for w in report.warnings)


class TestOverfitDetection:
    """Tests for overfitting detection."""

    def test_healthy_training_no_overfit(self):
        """Healthy training with gradual entropy decline (< 0.3/epoch per interval)."""
        # Each interval has drop < 0.3/epoch:
        # epoch 1→2: (1.1 - 0.9) / 1.0 = 0.2/epoch
        # epoch 2→3: (0.9 - 0.77) / 1.0 = 0.13/epoch
        log_history = [
            {"loss": 1.3, "entropy": 1.1, "mean_token_accuracy": 0.68, "epoch": 1.0},
            {"loss": 1.0, "entropy": 0.9, "mean_token_accuracy": 0.73, "epoch": 2.0},
            {"loss": 0.7, "entropy": 0.77, "mean_token_accuracy": 0.79, "epoch": 3.0},
        ]
        report = check_training_stability(log_history)
        assert report.stable is True
        assert report.warnings == []
        assert report.final_entropy == 0.77
        assert report.final_accuracy == 0.79

    def test_low_entropy_warning(self):
        """Warns when entropy drops below threshold (model too deterministic)."""
        log_history = [
            {"loss": 1.3, "entropy": 1.3, "epoch": 0.5},
            {"loss": 0.8, "entropy": 0.8, "epoch": 1.0},
            {"loss": 0.5, "entropy": 0.5, "epoch": 2.0},  # Below 0.6 threshold
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert any("Low entropy" in w for w in report.warnings)
        assert report.final_entropy == 0.5

    def test_near_zero_loss_warning(self):
        """Warns when loss approaches zero (memorization)."""
        log_history = [
            {"loss": 1.0, "entropy": 1.0, "epoch": 0.5},
            {"loss": 0.5, "entropy": 0.8, "epoch": 1.0},
            {"loss": 0.2, "entropy": 0.7, "epoch": 2.0},  # Below 0.3 threshold
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert any("Very low loss" in w for w in report.warnings)

    def test_high_accuracy_warning(self):
        """Warns when token accuracy exceeds threshold (memorization)."""
        log_history = [
            {"loss": 1.0, "mean_token_accuracy": 0.70, "entropy": 1.0, "epoch": 0.5},
            {"loss": 0.5, "mean_token_accuracy": 0.85, "entropy": 0.8, "epoch": 1.0},
            {"loss": 0.4, "mean_token_accuracy": 0.96, "entropy": 0.7, "epoch": 2.0},  # >95%
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert any("Very high token accuracy" in w for w in report.warnings)
        assert report.final_accuracy == 0.96

    def test_rapid_entropy_drop_warning(self):
        """Warns when entropy drops too fast per epoch."""
        log_history = [
            {"loss": 1.3, "entropy": 1.5, "epoch": 0.0},  # Start high
            {"loss": 0.5, "entropy": 0.7, "epoch": 2.0},  # Drop 0.4/epoch (> 0.3 threshold)
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        assert any("Rapid entropy drop" in w for w in report.warnings)

    def test_multiple_overfit_warnings(self):
        """Detects multiple overfitting indicators simultaneously."""
        log_history = [
            {"loss": 1.3, "entropy": 1.3, "mean_token_accuracy": 0.68, "epoch": 0.5},
            {"loss": 0.8, "entropy": 0.8, "mean_token_accuracy": 0.78, "epoch": 1.0},
            {"loss": 0.4, "entropy": 0.5, "mean_token_accuracy": 0.88, "epoch": 2.0},
            {"loss": 0.2, "entropy": 0.4, "mean_token_accuracy": 0.96, "epoch": 3.0},
        ]
        report = check_training_stability(log_history)
        assert report.stable is False
        # Should have all 4 overfit warnings
        assert any("Low entropy" in w for w in report.warnings)
        assert any("Very low loss" in w for w in report.warnings)
        assert any("Very high token accuracy" in w for w in report.warnings)
        assert any("Rapid entropy drop" in w for w in report.warnings)

    def test_missing_entropy_no_crash(self):
        """Handles logs without entropy field gracefully."""
        log_history = [
            {"loss": 1.0, "epoch": 0.5},
            {"loss": 0.8, "epoch": 1.0},
            {"loss": 0.6, "epoch": 2.0},
        ]
        report = check_training_stability(log_history)
        # Should not crash, just skip entropy checks
        assert report.final_entropy is None
        assert report.initial_entropy is None

    def test_misaligned_entropy_epoch_uses_paired_entries_only(self):
        """Uses only entries with both entropy AND epoch for drop rate calculation.

        Entries missing either value are excluded, ensuring the calculation
        uses correctly aligned data points.
        """
        # Entry 2 has entropy but no epoch - should be excluded from drop rate calc
        # Only entries 1 and 3 have both, so drop rate = (1.5 - 0.5) / 2.0 = 0.5/epoch
        log_history = [
            {"loss": 1.3, "entropy": 1.5, "epoch": 0.0},
            {"loss": 0.8, "entropy": 1.0},  # Missing epoch - excluded from pairs
            {"loss": 0.5, "entropy": 0.5, "epoch": 2.0},
        ]
        report = check_training_stability(log_history)
        # Should trigger rapid entropy drop (0.5/epoch > 0.3 threshold)
        assert any("Rapid entropy drop" in w for w in report.warnings)
        # Low entropy warning should also fire (independent check)
        assert any("Low entropy" in w for w in report.warnings)

    def test_same_length_different_rows_uses_aligned_pairs(self):
        """Ensures drop rate uses aligned pairs even when list lengths would match.

        When entropy and epoch appear in different rows (same count but different
        entries), only entries with BOTH values are used for the calculation.
        """
        # 2 entries with entropy only, 2 entries with epoch only
        # No entries have both -> no pairs -> no drop rate warning
        log_history = [
            {"loss": 1.3, "entropy": 1.5},  # Entropy only
            {"loss": 1.0, "epoch": 0.5},  # Epoch only
            {"loss": 0.8, "entropy": 0.4},  # Entropy only (would be "rapid" if misaligned)
            {"loss": 0.6, "epoch": 2.0},  # Epoch only
        ]
        report = check_training_stability(log_history)
        # No rapid entropy drop - no aligned pairs exist
        assert not any("Rapid entropy drop" in w for w in report.warnings)
        # Low entropy warning should still fire (independent check on entropies list)
        assert any("Low entropy" in w for w in report.warnings)

    def test_non_scalar_metrics_ignored(self):
        """Non-scalar metric values (dict/list) are gracefully ignored.

        Ensures the function doesn't crash on malformed log data and still
        computes warnings from valid scalar entries.
        """
        log_history = [
            {"loss": 1.3, "entropy": 1.1, "mean_token_accuracy": 0.68, "epoch": 1.0},
            # Non-scalar values that should be ignored
            {"loss": {"nested": "dict"}, "entropy": [1, 2, 3], "epoch": 2.0},
            {"loss": 0.8, "entropy": {"bad": "data"}, "mean_token_accuracy": [0.9], "epoch": 3.0},
            # Valid entry with low values to trigger warnings
            {"loss": 0.2, "entropy": 0.5, "mean_token_accuracy": 0.96, "epoch": 4.0},
        ]
        report = check_training_stability(log_history)
        # Should not crash, return valid report
        assert report is not None
        # Final values should be from last valid scalar entries
        assert report.final_entropy == 0.5
        assert report.final_accuracy == 0.96
        assert report.final_loss == 0.2
        # Warnings should fire based on valid scalar data
        assert any("Low entropy" in w for w in report.warnings)
        assert any("Very high token accuracy" in w for w in report.warnings)
        assert any("Very low loss" in w for w in report.warnings)
