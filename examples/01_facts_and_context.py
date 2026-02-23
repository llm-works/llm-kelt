#!/usr/bin/env python3
"""Example: Facts and Context Injection.

This example demonstrates:
1. Storing facts about a user/context
2. Building system prompts with injected facts
3. Using ContextQuery for context-aware LLM interactions

Prerequisites:
    - PostgreSQL database with pgvector extension
    - Config file at etc/llm-learn.yaml (or set LEARN_CONFIG_FILE env var)
    - LLM backend configured (local or OpenAI)

Usage:
    python examples/01_facts_and_context.py
"""

import asyncio
import sys
from pathlib import Path

# Allow running without package installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from _helpers import (
    CMD,
    H1,
    H2,
    INFO,
    LLM_A,
    LLM_Q,
    MUTED,
    OK,
    RESET,
    WARN,
    get_demo_context_key,
    psql_cmd,
)
from llm_infer.client import Factory as LLMClientFactory

from llm_learn import LearnClient, LearnClientFactory
from llm_learn.inference import ContextBuilder, ContextQuery


def setup_facts(learn: LearnClient):
    """Add sample facts for the demo."""
    print(f"\n{H2}▶ Adding Facts{RESET}")

    # Clear any existing facts for a clean demo
    existing = learn.atomic.assertions.list_active()
    if existing:
        for fact in existing:
            learn.atomic.assertions.deactivate(fact.id)
        print(f"  {MUTED}Cleared {len(existing)} existing facts{RESET}")

    # Add facts with specific internal conventions
    learn.atomic.assertions.add(
        "Error responses must use format: {error_code: 'ERR-XXX', message: str, request_id: str}",
        category="api",
    )
    learn.atomic.assertions.add(
        "All error codes start with ERR- followed by 3 digits (e.g., ERR-401, ERR-500)",
        category="api",
    )
    learn.atomic.assertions.add(
        "Every response must include the X-Request-ID header for tracing", category="api"
    )

    print(f"  {OK}✓ Added 3 facts:{RESET}")
    for fact in learn.atomic.assertions.list_active():
        print(f"    {MUTED}id={fact.id}{RESET} {INFO}[{fact.category}]{RESET} {fact.content}")

    print(
        f'\n  {CMD}▸ Verify:{RESET} {psql_cmd(learn)} -c "SELECT id, category, content '
        f"FROM memv1_facts WHERE context_key='{learn.context_key}' AND active=true;\""
    )


def demo_context_builder(learn: LearnClient):
    """Demonstrate building system prompts with injected facts."""
    print(f"\n{H2}▶ Building System Prompts{RESET}")

    context_builder = ContextBuilder(learn.atomic.assertions)

    base_prompt = "You are a coding assistant."
    system_prompt = context_builder.build_system_prompt(base_prompt)

    print(f'  {MUTED}Base prompt:{RESET} "{base_prompt}"')
    print(f"  {OK}✓ Generated{RESET} (facts auto-injected, grouped by category):")
    print(f"  {MUTED}{'─' * 56}{RESET}")
    for line in system_prompt.split("\n"):
        print(f"  {MUTED}│{RESET} {line}")
    print(f"  {MUTED}{'─' * 56}{RESET}")

    # Filter by category
    api_only = context_builder.build_system_prompt(
        base_prompt="You are a coding assistant.",
        categories=["api"],
    )
    print(f"\n  {OK}✓ Filtered{RESET} (only {INFO}'api'{RESET}):")
    print(f"  {MUTED}{'─' * 56}{RESET}")
    for line in api_only.split("\n"):
        print(f"  {MUTED}│{RESET} {line}")
    print(f"  {MUTED}{'─' * 56}{RESET}")

    return context_builder


async def demo_context_query(context_builder: ContextBuilder):
    """Demonstrate context-aware LLM interactions."""
    print(f"\n{H2}▶ LLM Queries: Without vs With Context{RESET}")

    try:
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory

        config = Config("etc/llm-learn.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
        llm_factory = LLMClientFactory(lg)
        llm_client = llm_factory.from_config(config.llm.to_dict())

        question = "Show me how to return an error from a Python API endpoint"

        # First: Query WITHOUT context injection
        print(f"\n  {MUTED}Without context (plain LLM query):{RESET}")
        print(f"  {LLM_Q}Q: {question}{RESET}")
        response_plain = await llm_client.chat_async(
            messages=[{"role": "user", "content": question}],
            system="You are a coding assistant.",
        )
        plain_clean = response_plain.content.replace("<think>\n\n</think>\n\n", "").strip()
        print(f"  {LLM_A}{plain_clean[:600]}{RESET}")
        if len(plain_clean) > 600:
            print(f"  {MUTED}[...truncated]{RESET}")

        # Second: Query WITH context injection
        print(f"\n  {OK}With context (facts injected into system prompt):{RESET}")
        print(f"  {LLM_Q}Q: {question}{RESET}")

        query = ContextQuery(
            client=llm_client,
            context_builder=context_builder,
            base_system_prompt="You are a coding assistant.",
        )
        response_ctx = await query.ask(question)
        ctx_clean = response_ctx.replace("<think>\n\n</think>\n\n", "").strip()
        print(f"  {LLM_A}{ctx_clean[:600]}{RESET}")
        if len(ctx_clean) > 600:
            print(f"  {MUTED}[...truncated]{RESET}")

        print(
            f"\n  {INFO}ℹ With context: response follows ERR-XXX format, includes request_id{RESET}"
        )

        await llm_client.aclose()

    except Exception as e:
        print(f"  {MUTED}[Skipped] No LLM backend: {type(e).__name__}{RESET}")
        print(
            f"  {MUTED}Start llm-infer or configure OpenAI in etc/llm-learn.yaml to enable.{RESET}"
        )


def demo_fact_management(learn: LearnClient):
    """Demonstrate fact management operations."""
    print(f"\n{H2}▶ Fact Management{RESET}")

    facts = learn.atomic.assertions.list_active()
    if not facts:
        print(f"  {MUTED}No facts to manage{RESET}")
        return

    # Update a fact
    fact = facts[0]
    old_content = fact.content
    new_content = "Error codes use format ERR-XXX where XXX is a 3-digit number"
    learn.atomic.assertions.update(fact.id, content=new_content)
    print(f"  {OK}✓ Updated{RESET} fact {MUTED}id={fact.id}{RESET}")
    print(f'    {MUTED}before:{RESET} "{old_content[:50]}{"..." if len(old_content) > 50 else ""}"')
    print(f'    {MUTED}after:{RESET}  "{new_content}"')

    # Get facts by category
    api_facts = learn.atomic.assertions.list_active(category="api")
    print(f"  {INFO}ℹ{RESET} Facts in {INFO}'api'{RESET} category: {len(api_facts)}")

    # Deactivate a fact (soft delete)
    learn.atomic.assertions.deactivate(fact.id)
    print(
        f"  {WARN}⚠ Deactivated{RESET} fact {MUTED}id={fact.id}{RESET} (soft-delete, still in DB)"
    )

    # Get statistics
    stats = learn.get_stats()
    atomic_stats = stats["atomic"]
    print(
        f"  {INFO}ℹ{RESET} Context stats: assertions={atomic_stats['assertions']}, "
        f"feedback={atomic_stats['feedback']}, content={stats['content']}"
    )

    print(
        f'\n  {CMD}▸ Verify:{RESET} {psql_cmd(learn)} -c "SELECT id, active, content '
        f'FROM memv1_facts WHERE id={fact.id};"'
    )


async def main():
    """Run the facts and context injection demo."""
    print(f"\n{H1}{'━' * 50}{RESET}")
    print(f"{H1}  Example 01: Facts and Context Injection{RESET}")
    print(f"{H1}{'━' * 50}{RESET}")

    # Initialize
    from appinfra.config import Config
    from appinfra.log import LogConfig, LoggerFactory

    from llm_learn import ClientContext

    config = Config("etc/llm-learn.yaml")
    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
    factory = LearnClientFactory(lg)

    # Create context for this example
    context_key = get_demo_context_key("example")
    context = ClientContext(context_key=context_key)
    learn = factory.create_from_config(context=context, config=config)
    print(f"{MUTED}Using context_key={RESET}{INFO}{context_key}{RESET}")

    # Run demos
    setup_facts(learn)
    context_builder = demo_context_builder(learn)
    await demo_context_query(context_builder)
    demo_fact_management(learn)

    print(f"\n{H1}{'━' * 50}{RESET}")
    print(f"{OK}✓ Done!{RESET} Next: {CMD}python examples/02_rag_retrieval.py{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
