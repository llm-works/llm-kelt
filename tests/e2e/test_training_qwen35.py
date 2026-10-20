# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""E2E training on Qwen3.5-architecture checkpoints: SFT, DPO and prompt tuning.

Runs on tiny random models (see tiny_qwen35.py), bf16 and compressed-tensors W4A16, so it fits
any small GPU. Requires the local qwen2.5-0.5b-instruct (for its tokenizer).
"""

import json
from pathlib import Path

import pytest

SFT_RECORDS = [{"instruction": f"Say the word foo{i}.", "output": f"foo{i}"} for i in range(8)]
DPO_RECORDS = [
    {"prompt": f"Say the word foo{i}.", "chosen": f"foo{i}", "rejected": "bar"} for i in range(8)
]

# 4 layers x MLP (3) + 1 full-attention layer x q/k/v/o (4) + 3 linear-attention layers x
# in_proj_qkv/in_proj_z/out_proj (3)
EXPECTED_LORA_MODULES = 4 * 3 + 4 + 3 * 3


@pytest.fixture(params=["bf16", "compressed"])
def qwen35_model(request) -> Path:
    fixture = "tiny_qwen35_bf16" if request.param == "bf16" else "tiny_qwen35_compressed"
    return request.getfixturevalue(fixture)


@pytest.fixture
def train_config():
    import torch
    from appinfra import DotDict

    return DotDict(
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=64,
        logging_steps=1,
        save_steps=1000,
        gradient_checkpointing=True,
        eval_split=0.0,
        learning_rate=1e-3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        lora_profile="small",
    )


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _losses(trainer) -> list[float]:
    return [h["loss"] for h in trainer.trainer.state.log_history if "loss" in h]


def _lora_module_names(model) -> list[str]:
    return [name for name, module in model.named_modules() if hasattr(module, "lora_A")]


def _assert_language_model_lora(model) -> None:
    names = _lora_module_names(model)
    assert len(names) == EXPECTED_LORA_MODULES
    leaves = {n.rsplit(".", 1)[-1] for n in names}
    assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "down_proj"} <= leaves
    assert not [n for n in names if ".visual." in n]


@pytest.mark.training
class TestQwen35Training:
    def test_sft(self, logger, qwen35_model: Path, train_config, tmp_path: Path):
        from llm_kelt.training.lora.trainer import Trainer

        trainer = Trainer(
            lg=logger,
            data_path=_write(tmp_path / "sft.jsonl", SFT_RECORDS),
            output_dir=tmp_path / "out",
            base_model=str(qwen35_model),
            training_config=train_config,
            quantize=False,
        )
        result = trainer.train()

        assert type(trainer.model.get_base_model()).__name__ == "Qwen3_5ForConditionalGeneration"
        _assert_language_model_lora(trainer.model)
        losses = _losses(trainer)
        assert losses[-1] < losses[0]
        assert "in_proj_qkv" in result.config["lora"]["target_modules"]
        assert (Path(result.adapter.path) / "adapter_model.safetensors").exists()

    def test_dpo(self, logger, qwen35_model: Path, train_config, tmp_path: Path):
        from llm_kelt.training.dpo.trainer import Trainer

        trainer = Trainer(
            lg=logger,
            data_path=_write(tmp_path / "dpo.jsonl", DPO_RECORDS),
            output_dir=tmp_path / "out",
            base_model=str(qwen35_model),
            training_config=train_config,
            quantize=False,
        )
        result = trainer.train()

        assert result.status == "completed"
        _assert_language_model_lora(trainer.model)
        losses = _losses(trainer)
        assert losses[-1] < losses[0]

    def test_prompt_tuning(self, logger, qwen35_model: Path, train_config, tmp_path: Path):
        from llm_kelt.training.prompt import Config as PromptConfig
        from llm_kelt.training.prompt import Trainer

        trainer = Trainer(
            lg=logger,
            data_path=_write(tmp_path / "sft.jsonl", SFT_RECORDS),
            output_dir=tmp_path / "out",
            base_model=str(qwen35_model),
            prompt_config=PromptConfig(
                num_virtual_tokens=4,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text="Say the word.",
            ),
            training_config=train_config,
            quantize=False,
        )
        result = trainer.train()

        assert result.method == "prompt"
        losses = _losses(trainer)
        assert losses[-1] < losses[0]


@pytest.mark.training
def test_compressed_checkpoint_loads_with_source_weights(
    logger, tiny_qwen35_bf16: Path, tiny_qwen35_compressed: Path
):
    """Unquantized layers load exactly; int4 layers load close to the bf16 source."""
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    from llm_kelt.training.model import load_base_model

    model = load_base_model(logger, str(tiny_qwen35_compressed), None)
    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        tiny_qwen35_bf16, dtype=torch.bfloat16
    ).state_dict()
    loaded = model.state_dict()

    def rel_err(name: str) -> float:
        a, b = loaded[name].float().cpu(), source[name].float()
        return ((a - b).norm() / b.norm()).item()

    prefix = "model.language_model.layers"
    assert rel_err(f"{prefix}.0.linear_attn.in_proj_qkv.weight") == 0.0
    assert rel_err(f"{prefix}.0.linear_attn.out_proj.weight") == 0.0
    # Symmetric int4 with group size 128: ~0.11 relative error on random normal weights
    assert 0.0 < rel_err(f"{prefix}.3.self_attn.q_proj.weight") < 0.2
    assert 0.0 < rel_err(f"{prefix}.0.mlp.down_proj.weight") < 0.2
