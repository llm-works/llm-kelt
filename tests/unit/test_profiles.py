"""Tests for model-size-aware training profiles."""

import pytest

from llm_kelt.training.profiles import (
    MODEL_SIZE_PROFILES,
    ProfileDetectionError,
    _extract_size_from_name,
    get_model_size_profile,
    get_size_profile_name,
)


class TestExtractSizeFromName:
    """Tests for _extract_size_from_name (fallback parser)."""

    def test_simple_7b(self):
        assert _extract_size_from_name("Qwen2.5-7B-Instruct") == 7.0

    def test_simple_32b(self):
        assert _extract_size_from_name("Qwen2.5-32B-Instruct") == 32.0

    def test_with_path(self):
        assert _extract_size_from_name("Qwen/Qwen2.5-14B-Instruct") == 14.0

    def test_lowercase_b(self):
        assert _extract_size_from_name("llama-70b-chat") == 70.0

    def test_decimal_size(self):
        assert _extract_size_from_name("Qwen2.5-1.5B") == 1.5

    def test_with_quantization_suffix(self):
        assert _extract_size_from_name("qwen2.5-32b-instruct-bnb-4bit") == 32.0

    def test_no_size_returns_none(self):
        assert _extract_size_from_name("gpt2") is None

    def test_b_in_name_not_size(self):
        # "BNB" should not match as size
        assert _extract_size_from_name("some-model-BNB") is None


class TestGetSizeProfileName:
    """Tests for get_size_profile_name."""

    def test_small_7b(self):
        assert get_size_profile_name(7.0) == "small"

    def test_small_14b(self):
        assert get_size_profile_name(14.0) == "small"

    def test_medium_32b(self):
        assert get_size_profile_name(32.0) == "medium"

    def test_medium_50b(self):
        assert get_size_profile_name(50.0) == "medium"

    def test_large_70b(self):
        assert get_size_profile_name(70.0) == "large"

    def test_xlarge_72b(self):
        assert get_size_profile_name(72.0) == "xlarge"

    def test_xlarge_405b(self):
        assert get_size_profile_name(405.0) == "xlarge"


class TestGetModelSizeProfile:
    """Tests for get_model_size_profile."""

    def test_small_model(self):
        name, profile = get_model_size_profile("Qwen/Qwen2.5-7B-Instruct")
        assert name == "small"
        assert profile["lora"]["r"] == 16
        assert profile["lora"]["lora_alpha"] == 32
        assert profile["training"]["learning_rate"] == 0.0002

    def test_medium_model(self):
        name, profile = get_model_size_profile("Qwen/Qwen2.5-32B-Instruct")
        assert name == "medium"
        assert profile["lora"]["r"] == 64
        assert profile["lora"]["lora_alpha"] == 128
        assert profile["training"]["learning_rate"] == 0.0001

    def test_large_model(self):
        name, profile = get_model_size_profile("meta-llama/Llama-3-70B-Instruct")
        assert name == "large"
        assert profile["lora"]["r"] == 128
        assert profile["lora"]["lora_alpha"] == 256
        assert profile["training"]["learning_rate"] == 0.00005

    def test_xlarge_model(self):
        name, profile = get_model_size_profile("Qwen/Qwen2.5-72B-Instruct")
        assert name == "xlarge"
        assert profile["lora"]["r"] == 128
        assert profile["lora"]["lora_alpha"] == 256
        assert profile["training"]["learning_rate"] == 0.00002

    def test_unknown_size_falls_back_to_small(self):
        name, profile = get_model_size_profile("gpt2")
        assert name == "small"
        assert profile == MODEL_SIZE_PROFILES["small"]

    def test_profile_override_skips_detection(self):
        # gpt2 would fall back to small, but we force large
        name, profile = get_model_size_profile("gpt2", profile_override="large")
        assert name == "large"
        assert profile["lora"]["r"] == 128

    def test_require_detection_raises_on_unknown(self):
        with pytest.raises(ProfileDetectionError) as exc_info:
            get_model_size_profile("gpt2", require_detection=True)
        assert "Cannot detect model size" in str(exc_info.value)
        assert "--lora-profile" in str(exc_info.value)

    def test_require_detection_ok_with_override(self):
        # Should not raise even with require_detection if override given
        name, profile = get_model_size_profile(
            "gpt2", profile_override="medium", require_detection=True
        )
        assert name == "medium"
