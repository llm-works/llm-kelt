#!/usr/bin/env python3
"""Example: Exporting Data for Training.

This example demonstrates:
1. Recording feedback and preferences
2. Exporting to DPO format (preference pairs)
3. Exporting to SFT format (supervised fine-tuning)
4. Exporting to classifier format (binary classification)

Prerequisites:
    - PostgreSQL database with pgvector extension
    - Config file at etc/llm-learn.yaml

Usage:
    python examples/03_training_export.py
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

# Allow running without package installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from _helpers import CMD, H1, H2, INFO, MUTED, OK, RESET, psql_cmd

from llm_learn import IsolationContext, LearnClient, LearnClientFactory
from llm_learn.training import (
    ExportResult,
    export_feedback_classifier,
    export_feedback_sft,
    export_preferences_dpo,
)

# Sample content: (text, title)
_SAMPLE_CONTENT = [
    (
        "Machine learning is a subset of AI that enables systems to learn "
        "from data without being explicitly programmed.",
        "What is machine learning?",
    ),
    (
        "Neural networks are computing systems inspired by biological neural "
        "networks. They consist of layers of interconnected nodes.",
        "Explain neural networks",
    ),
    (
        "Gradient descent is an optimization algorithm used to minimize the "
        "loss function in machine learning models.",
        "What is gradient descent?",
    ),
    (
        "Overfitting occurs when a model learns the training data too well, "
        "including noise, and fails to generalize to new data.",
        "What is overfitting?",
    ),
    (
        "This response is too verbose and doesn't get to the point. "
        "It contains unnecessary information.",
        "Bad response example",
    ),
]

# Sample preference pairs: (context, chosen, rejected)
_PREFERENCE_PAIRS = [
    (
        "Explain backpropagation",
        "Backpropagation computes gradients by applying the chain rule backward through layers.",
        "Backpropagation is a really complex algorithm that does a lot of mathematical computations "
        "involving derivatives and chains and lots of other complicated stuff.",
    ),
    (
        "What is a learning rate?",
        "Learning rate controls step size during gradient descent. Too high causes divergence.",
        "The learning rate is a hyperparameter. It's a number. You set it before training.",
    ),
    (
        "Explain regularization",
        "Regularization prevents overfitting by adding a penalty term to the loss function.",
        "Regularization is when you add stuff to your loss function to make your model work better.",
    ),
]


def print_export_result(result: ExportResult, show_content: bool = True):
    """Pretty print export result."""
    print(f"    {MUTED}Path:{RESET} {result.path}")
    print(f"    {MUTED}Count:{RESET} {OK}{result.count}{RESET}")
    print(f"    {MUTED}Format:{RESET} {result.format}")

    if show_content and result.count > 0:
        print(f"    {MUTED}Sample:{RESET}")
        with open(result.path) as f:
            for i, line in enumerate(f):
                if i >= 2:
                    print(f"      {MUTED}...{RESET}")
                    break
                record = json.loads(line)
                display = {
                    k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
                    for k, v in record.items()
                }
                print(f"      {MUTED}{display}{RESET}")


def record_sample_data(learn: LearnClient):
    """Record sample feedback and preferences for the demo."""
    print(f"\n{H2}▶ Recording Sample Data{RESET}")

    # Clear existing data for clean demo
    from llm_learn.core.models import Content
    from llm_learn.memory.atomic.models import Fact, FeedbackDetails, PreferenceDetails

    with learn.database.session() as session:
        session.query(FeedbackDetails).filter(
            FeedbackDetails.fact_id.in_(
                session.query(Fact.id).filter_by(context_key=learn.context_key, type="feedback")
            )
        ).delete(synchronize_session=False)
        session.query(PreferenceDetails).filter(
            PreferenceDetails.fact_id.in_(
                session.query(Fact.id).filter_by(context_key=learn.context_key, type="preference")
            )
        ).delete(synchronize_session=False)
        session.query(Fact).filter_by(context_key=learn.context_key).delete()
        session.query(Content).filter_by(context_key=learn.context_key).delete()
        session.commit()
    print(f"  {MUTED}Cleared existing data for clean demo{RESET}")

    # Create content and record feedback
    content_ids = []
    for text, title in _SAMPLE_CONTENT:
        cid = learn.content.create(content_text=text, source="example", title=title)
        content_ids.append(cid)

    # Positive feedback for good responses
    for cid in content_ids[:4]:
        learn.feedback.record(
            signal="positive", content_id=cid, strength=0.9, tags=["clear", "concise"]
        )

    # Negative feedback for bad response
    learn.feedback.record(
        signal="negative", content_id=content_ids[4], strength=0.8, tags=["verbose", "unclear"]
    )

    print(f"  {OK}✓ Recorded {learn.feedback.count()} feedback entries{RESET}")

    # Record preference pairs
    for context, chosen, rejected in _PREFERENCE_PAIRS:
        learn.preferences.record(
            context=context,
            chosen=chosen,
            rejected=rejected,
            category="ml_explanations",
            margin=0.8,
        )

    print(f"  {OK}✓ Recorded {learn.preferences.count()} preference pairs{RESET}")

    print(
        f'\n  {CMD}▸ Verify feedback:{RESET} {psql_cmd(learn)} -c "SELECT f.id, d.signal, d.strength '
        f"FROM memv1_facts f JOIN memv1_feedback_details d ON f.id = d.fact_id "
        f'WHERE f.context_key={learn.context_key} LIMIT 5;"'
    )
    print(
        f'  {CMD}▸ Verify preferences:{RESET} {psql_cmd(learn)} -c "SELECT f.id, f.category, d.margin '
        f"FROM memv1_facts f JOIN memv1_preference_details d ON f.id = d.fact_id "
        f'WHERE f.context_key={learn.context_key} LIMIT 5;"'
    )


def demo_dpo_export(output_dir: Path, learn: LearnClient):
    """Export to DPO format."""
    print(f"\n{H2}▶ DPO Export{RESET}")
    print(f"  {MUTED}Format: {{prompt, chosen, rejected}}{RESET}")
    print(f"  {MUTED}Use case: TRL DPOTrainer for preference alignment{RESET}")

    result = export_preferences_dpo(
        session_factory=learn.database.session,
        context_key=learn.context_key,
        output_path=output_dir / "preferences_dpo.jsonl",
        category="ml_explanations",
    )
    print_export_result(result)


def demo_sft_export(output_dir: Path, learn: LearnClient):
    """Export to SFT format."""
    print(f"\n{H2}▶ SFT Export{RESET}")
    print(f"  {MUTED}Format: {{instruction, output}} (Alpaca format){RESET}")
    print(f"  {MUTED}Use case: Supervised fine-tuning on positive examples{RESET}")

    result = export_feedback_sft(
        session_factory=learn.database.session,
        context_key=learn.context_key,
        output_path=output_dir / "feedback_sft.jsonl",
        signal="positive",
        min_strength=0.5,
    )
    print_export_result(result)

    # SFT with context
    print(f"\n  {INFO}With context (includes input field):{RESET}")
    result_ctx = export_feedback_sft(
        session_factory=learn.database.session,
        context_key=learn.context_key,
        output_path=output_dir / "feedback_sft_context.jsonl",
        signal="positive",
        include_context=True,
    )
    print_export_result(result_ctx)


def demo_classifier_export(output_dir: Path, learn: LearnClient):
    """Export to classifier format."""
    print(f"\n{H2}▶ Classifier Export{RESET}")
    print(f"  {MUTED}Format: {{text, label}}{RESET}")
    print(f"  {MUTED}Use case: Binary quality classifier (good/bad responses){RESET}")

    result = export_feedback_classifier(
        session_factory=learn.database.session,
        context_key=learn.context_key,
        output_path=output_dir / "feedback_classifier.jsonl",
        min_strength=0.5,
    )
    print_export_result(result)


def demo_filtered_export(output_dir: Path, learn: LearnClient):
    """Export with time filters."""
    print(f"\n{H2}▶ Filtered Export{RESET}")
    print(f"  {MUTED}Filter exports by time range{RESET}")

    since = datetime.now(UTC) - timedelta(hours=1)
    result = export_preferences_dpo(
        session_factory=learn.database.session,
        context_key=learn.context_key,
        output_path=output_dir / "recent_preferences.jsonl",
        since=since,
    )
    print(f"  {INFO}Recent preferences (since {since.strftime('%H:%M:%S')}):{RESET}")
    print_export_result(result, show_content=False)


def print_summary():
    """Print export function summary."""
    print(f"\n{H2}▶ Export Functions Summary{RESET}")
    print(f"""
  {INFO}export_preferences_dpo(){RESET}
    Output: {{prompt, chosen, rejected}}
    Filters: domain, since, until, min_margin

  {INFO}export_feedback_sft(){RESET}
    Output: {{instruction, output}} or {{instruction, input, output}}
    Filters: signal, min_strength, since, until

  {INFO}export_feedback_classifier(){RESET}
    Output: {{text, label}}
    Filters: since, until, min_strength
""")


def main():
    """Run the training export demo."""
    from appinfra.config import Config
    from appinfra.log import LogConfig, LoggerFactory

    print(f"\n{H1}{'━' * 50}{RESET}")
    print(f"{H1}  Example 03: Training Data Export{RESET}")
    print(f"{H1}{'━' * 50}{RESET}")

    # Setup config, logger, and database
    config = Config("etc/llm-learn.yaml")
    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))

    # Create LearnClient using factory
    context = IsolationContext(context_key="demo:example")
    factory = LearnClientFactory(lg)
    learn = factory.create_from_config(context=context, config=config)
    print(f"{MUTED}Using context_key={RESET}{INFO}{learn.context_key}{RESET}")

    record_sample_data(learn)

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        demo_dpo_export(output_dir, learn)
        demo_sft_export(output_dir, learn)
        demo_classifier_export(output_dir, learn)
        demo_filtered_export(output_dir, learn)

    print_summary()

    print(f"\n{H1}{'━' * 50}{RESET}")
    print(f"{OK}✓ Done!{RESET} Next: {CMD}python examples/04_lora_training.py{RESET}")


if __name__ == "__main__":
    main()
