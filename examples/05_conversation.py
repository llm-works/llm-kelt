#!/usr/bin/env python3
"""Example: Conversation Layer.

This example demonstrates:
1. Creating conversations with token tracking
2. Compaction (sliding window)
3. Persisting sessions to file and loading them back
4. Multi-turn LLM conversation with session persistence (optional, requires LLM backend)

Prerequisites:
    - No external dependencies for demos 1-3
    - LLM backend for demo 4 (configure in etc/llm-kelt.yaml)

Usage:
    python examples/05_conversation.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Allow running without package installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from _helpers import H1, H2, INFO, LLM_A, LLM_Q, MUTED, OK, RESET, WARN
from appinfra.log import LogConfig, LoggerFactory

from llm_kelt.conversation import (
    Config,
    Conversation,
    Message,
    Role,
    SlidingWindowCompactor,
    ToolCall,
)
from llm_kelt.conversation.storage import FileSessionStorage


def demo_types():
    """Demonstrate Message and ToolCall types."""
    print(f"\n{H1}{'=' * 60}")
    print("  Conversation Layer Demo")
    print(f"{'=' * 60}{RESET}")

    print(f"\n{H2}▶ Message Types{RESET}")

    # Basic messages
    user_msg = Message(role="user", content="What files are in the current directory?")
    print(f"  {INFO}User message:{RESET} {user_msg.role}: {user_msg.content}")

    # Assistant with tool calls
    assistant_msg = Message(
        role="assistant",
        content="",
        tool_calls=[dict(ToolCall(id="tc_1", name="list_files", arguments={"path": "."}))],
    )
    print(f"  {INFO}Assistant tool call:{RESET} {assistant_msg.tool_calls}")

    # Tool result
    tool_msg = Message(role="tool", content="main.py\nutils.py\nREADME.md", tool_call_id="tc_1")
    print(f"  {INFO}Tool result:{RESET} {tool_msg.content}")

    # Messages are dicts (FieldDict)
    d = dict(user_msg)
    print(f"  {MUTED}As dict: {d}{RESET}")


def demo_conversation():
    """Demonstrate conversation management with token tracking."""
    print(f"\n{H2}▶ Conversation Management{RESET}")

    config = Config(max_tokens=200, compact_threshold=0.8, min_recent_messages=2)
    conv = Conversation(config=config)

    conv.add("You are a helpful coding assistant.", Role.SYSTEM)
    conv.add("What is a Python decorator?")
    conv.add(
        "A decorator is a function that wraps another function to extend its behavior "
        "without modifying its code. You use the @syntax to apply them.",
        Role.ASSISTANT,
    )
    conv.add("Can you show me an example?")
    conv.add(
        "Sure! Here's a simple timing decorator:\n\n"
        "def timer(func):\n"
        "    def wrapper(*args):\n"
        "        start = time.time()\n"
        "        result = func(*args)\n"
        "        print(f'Took {time.time() - start:.2f}s')\n"
        "        return result\n"
        "    return wrapper",
        Role.ASSISTANT,
    )

    print(f"  {INFO}Messages:{RESET} {conv.message_count}")
    print(f"  {INFO}Token count:{RESET} {conv.token_count}")
    print(f"  {INFO}Token limit:{RESET} {conv.token_limit}")
    print(f"  {INFO}Usage ratio:{RESET} {conv.usage_ratio:.1%}")
    print(f"  {INFO}Needs compaction:{RESET} {conv.needs_compaction()}")

    return conv


def demo_compaction(conv: Conversation):
    """Demonstrate sliding window compaction."""
    print(f"\n{H2}▶ Compaction{RESET}")

    if not conv.needs_compaction():
        print(f"  {MUTED}(Adding more messages to trigger compaction...){RESET}")
        conv.add("What about class decorators?")
        conv.add("Class decorators work similarly but wrap entire classes...", Role.ASSISTANT)

    print(f"  {WARN}Before:{RESET} {conv.message_count} messages, {conv.token_count} tokens")
    print(f"  {WARN}Needs compaction:{RESET} {conv.needs_compaction()}")

    compactor = SlidingWindowCompactor()
    compactor.compact(conv)

    print(f"  {OK}After:{RESET}  {conv.message_count} messages, {conv.token_count} tokens")
    print(f"  {MUTED}Preserved messages:{RESET}")
    for msg in conv.messages:
        role = msg.role.upper()
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"    [{role}] {preview}")


def demo_storage():
    """Demonstrate file-based session persistence."""
    print(f"\n{H2}▶ Session Storage (File Backend){RESET}")

    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileSessionStorage(lg, tmpdir)

        # Create and save sessions
        for i, (question, answer) in enumerate(
            [
                ("How do I read a file in Python?", "Use open() with a context manager..."),
                ("What's the difference between list and tuple?", "Lists are mutable, tuples..."),
                ("Explain async/await", "async/await enables concurrent execution..."),
            ],
            start=1,
        ):
            conv = Conversation()
            conv.add(question)
            conv.add(answer, Role.ASSISTANT)
            storage.save(f"session-{i}", conv, metadata={"model": "qwen2.5-7b"})
            print(f"  {OK}Saved:{RESET} session-{i}")

        # List sessions
        print(f"\n  {INFO}Listing sessions:{RESET}")
        for summary in storage.list():
            print(
                f"    {summary.session_id}  "
                f"msgs={summary.message_count}  "
                f"tokens={summary.token_count}  "
                f'preview="{summary.preview}"'
            )

        # Load and inspect
        print(f"\n  {INFO}Loading session-1:{RESET}")
        loaded = storage.load("session-1")
        print(f"    Session ID: {loaded.session_id}")
        print(f"    Messages: {len(loaded.messages)}")
        print(f"    Created: {loaded.created_at}")
        print(f"    Metadata: {loaded.metadata}")

        # Delete
        storage.delete("session-2")
        print(f"\n  {WARN}Deleted session-2{RESET}")
        remaining = storage.list()
        print(f"  {INFO}Remaining sessions:{RESET} {len(remaining)}")


async def demo_llm_conversation():
    """Demonstrate multi-turn LLM conversation with session persistence.

    Requires an LLM backend configured in etc/llm-kelt.yaml.
    Skips gracefully if unavailable.
    """
    print(f"\n{H2}▶ Multi-turn LLM Conversation{RESET}")

    try:
        from appinfra.config import Config as AppConfig
        from llm_infer.client import Factory as LLMClientFactory

        from llm_kelt.inference.query import ContextQuery
    except ImportError:
        print(f"  {MUTED}(Skipped: llm-infer not installed){RESET}")
        return

    config_path = Path(__file__).parent.parent / "etc" / "llm-kelt.yaml"
    if not config_path.exists():
        print(f"  {MUTED}(Skipped: {config_path} not found){RESET}")
        return

    config = AppConfig(str(config_path))
    llm_config = getattr(config, "llm", None)
    if not llm_config:
        print(f"  {MUTED}(Skipped: no llm section in config){RESET}")
        return

    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))

    try:
        llm_factory = LLMClientFactory(lg)
        llm_client = llm_factory.from_config(llm_config.to_dict())
    except Exception as e:
        print(f"  {MUTED}(Skipped: failed to create LLM client: {e}){RESET}")
        return

    # Create a mock context builder (no DB needed for this demo)
    from unittest.mock import MagicMock

    mock_builder = MagicMock()
    mock_builder.build_system_prompt.return_value = (
        "You are a knowledgeable Python tutor. Keep answers concise (2-3 sentences)."
    )

    query = ContextQuery(
        client=llm_client,
        context_builder=mock_builder,
        base_system_prompt="You are a knowledgeable Python tutor. Keep answers concise.",
    )

    conv = Conversation(config=Config(max_tokens=4000))
    questions = [
        "What are Python generators?",
        "How do they differ from list comprehensions?",
        "When should I prefer one over the other?",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileSessionStorage(lg, tmpdir)

        try:
            for question in questions:
                print(f"\n  {LLM_Q}User:{RESET} {question}")
                response = await query.ask(question, conversation=conv)
                print(f"  {LLM_A}Assistant:{RESET} {response}")

                # Persist after each turn
                storage.save("llm-demo", conv, metadata={"model": "local"})

            print(f"\n  {INFO}Session stats:{RESET}")
            print(f"    Messages: {conv.message_count}")
            print(f"    Tokens: {conv.token_count}")
            print(f"    Usage: {conv.usage_ratio:.1%}")

            # Verify persistence
            loaded = storage.load("llm-demo")
            print(f"    Persisted messages: {len(loaded.messages)}")

        except Exception as e:
            print(f"  {WARN}LLM call failed: {e}{RESET}")
            print(f"  {MUTED}(Is the LLM backend running?){RESET}")
        finally:
            await query.close()


if __name__ == "__main__":
    demo_types()
    conv = demo_conversation()
    demo_compaction(conv)
    demo_storage()
    asyncio.run(demo_llm_conversation())

    print(f"\n{H1}{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}{RESET}\n")
