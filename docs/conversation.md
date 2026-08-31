# Conversation

Multi-turn dialogue container with token accounting, compaction, and file-backed persistence.
Separate subsystem from atomic memory — plug together as needed.

## Imports

```python
from llm_kelt.conversation import (
    Conversation,
    Config,
    Message,
    Role,
    ToolCall,
    SlidingWindowCompactor,
    SummarizingCompactor,
    TieredCompactor,
    AsyncTieredCompactor,
    ContextOverflowError,
)
from llm_kelt.conversation.compaction import (
    CompactionGuardError,
    max_summary_tokens,
    preserve_keywords,
    token_reduction,
)
from llm_kelt.conversation.storage import FileSessionStorage
```

## `Message`, `Role`, `ToolCall`

Plain dataclasses re-exported from `llm-saia`:

```python
Message(role: str, content: str, tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None)

Role.USER | Role.ASSISTANT | Role.SYSTEM | Role.TOOL   # str-valued enum

ToolCall(id: str, name: str, arguments: dict)
```

Assistant tool calls and tool results are just messages with the appropriate role:

```python
Message(
    role="assistant",
    content="",
    tool_calls=[ToolCall(id="tc_1", name="list_files", arguments={"path": "."})],
)

Message(role="tool", content="main.py\nutils.py", tool_call_id="tc_1")
```

## `Config`

```python
class Config(FieldDict):
    max_tokens: int = 32000
    compact_threshold: float = 0.8  # trigger point as fraction of max_tokens
    preserve_system: bool = True
    min_recent_messages: int = 4  # never compact away the last N
    tokenizer: Tokenizer | None = None  # callable str → int
```

`Tokenizer` is `Callable[[str], int]`. Leave as `None` to use the built-in char-based
estimate (`estimate_tokens`), which is fast and good enough for header room. Supply a real
tokenizer (`tiktoken`, HF fast tokenizer) when you need precise budgeting.

## `Conversation`

```python
class Conversation:
    def __init__(
        self,
        lg: Logger,
        config: Config | None = None,
        compactor: Compactor | AsyncCompactor | None = None,
    )

    # ConversationLike protocol
    def append(self, msg: Message) -> None
    async def append_async(self, msg: Message) -> None
    def as_messages(self) -> list[Message]           # live view

    # Convenience
    def add(
        self,
        content: str,
        role: str | Role = Role.USER,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None,
    ) -> None

    # Views
    messages -> list[Message]                        # copy
    messages_as_dicts() -> list[dict]
    message_count -> int
    token_count -> int
    token_limit -> int
    usage_ratio -> float
    needs_compaction() -> bool

    # Mutation
    clear() -> None
    replace_messages(messages: list[Message]) -> None
    get_system_message() -> Message | None
    split_for_compaction() -> tuple[list[Message], list[Message]]  # (compactable, preserved)

    # Persistence
    to_dict() -> dict
    @classmethod
    def from_dict(cls, data, lg, config=None, compactor=None) -> Self
```

`append` runs the sync compactor synchronously. If you constructed with an `AsyncCompactor`,
`append` raises `RuntimeError` — use `append_async` instead.

`ContextOverflowError` is raised from `append` when the new message can't fit even after
running the configured compactor. Catch it, drop or summarise, retry.

Basic use:

```python
conv = Conversation(lg, config=Config(max_tokens=8000, compact_threshold=0.75))

conv.add("You are a helpful coding assistant.", Role.SYSTEM)
conv.add("What's a Python decorator?")
conv.add("A decorator wraps a function to extend its behaviour...", Role.ASSISTANT)

print(conv.message_count, conv.token_count, conv.usage_ratio)
```

## Compactors

Four compactors, all interchangeable:

| Compactor | Style | Async? | Notes |
|---|---|---|---|
| `SlidingWindowCompactor` | Drop oldest | Sync | Cheapest; loses context |
| `SummarizingCompactor` | LLM-generated summary | Async | Preserves gist; costs a call per compaction |
| `TieredCompactor` | Structured rewrite (trim tool results, drop old turns) | Sync | Middle ground |
| `AsyncTieredCompactor` | Same tiered strategy, async | Async | For async pipelines |

### `SlidingWindowCompactor`

```python
conv = Conversation(lg, config=cfg, compactor=SlidingWindowCompactor())
```

On each `append`, if `usage_ratio > compact_threshold`, drop the oldest messages until under
threshold. Never drops the system message (with `preserve_system=True`) or the last
`min_recent_messages` turns.

### `SummarizingCompactor`

Uses an async LLM client to summarise the compactable prefix into a single synthetic
`assistant` message:

```python
from llm_infer.client import Factory as LLMFactory

llm_client = LLMFactory(lg).from_config(config.llm.to_dict())

compactor = SummarizingCompactor(
    client=llm_client,
    guards=[
        token_reduction(min_ratio=0.5),  # summary must be ≤50% of input tokens
        preserve_keywords(["ERR-401", "budget"]),  # summary must mention these strings
        max_summary_tokens(1500),  # summary itself capped
    ],
)

conv = Conversation(lg, config=cfg, compactor=compactor)
await conv.append_async(msg)
```

If any guard fails, `CompactionGuardError` is raised and the conversation is left
uncompacted. Catch and decide: fall back to sliding window, drop, or surface.

Guards:

```python
token_reduction(min_ratio: float)      # summary tokens / compactable tokens ≤ 1 - min_ratio
preserve_keywords(keywords: list[str]) # every keyword must appear in summary
max_summary_tokens(limit: int)         # cap on summary length
```

Combine as many as needed — all must pass.

### `TieredCompactor` / `AsyncTieredCompactor`

Tiered strategy: first trim known-large tool results (`DEFAULT_TRIMMABLE_TOOLS` constant lists
the tool names to shorten), then drop oldest turns if still over budget. Preserves the
message *structure* — no synthetic assistant message inserted.

```python
compactor = TieredCompactor(
    client=llm_client,
    trimmable_tools=["read_file", "grep", "list_dir"],
)
conv = Conversation(lg, config=cfg, compactor=compactor)
```

## Token accounting

```python
from llm_kelt.conversation import estimate_tokens, estimate_message_tokens

# Char-based estimate (fast, rough)
estimate_tokens("hello world")  # → 3

# With a real tokenizer
from tiktoken import encoding_for_model

enc = encoding_for_model("gpt-4o")
estimate_tokens("hello world", tokenizer=lambda s: len(enc.encode(s)))

# Full message including role and tool-call overhead
estimate_message_tokens(role="assistant", content="…", tool_calls=[...])
```

Passing your tokenizer to `Config(tokenizer=…)` makes `conv.token_count` use it for every
message.

## Storage

`FileSessionStorage` persists conversations to a directory as JSON, one file per session:

```python
class FileSessionStorage(SessionStorage):
    def __init__(self, lg: Logger, base_path: str | Path)

    def save(self, session_id: str, conversation: Conversation,
             extra: dict | None = None) -> None
    def load(self, session_id: str) -> StoredSession
    def list(self, limit: int = 20) -> list[SessionSummary]
    def delete(self, session_id: str) -> bool
```

`StoredSession` fields: `session_id`, `messages`, `created_at`, `updated_at`, `extra`.
`SessionSummary`: `session_id`, `message_count`, `token_count`, `preview`, `updated_at`.

Example:

```python
storage = FileSessionStorage(lg, "~/.llm-kelt/sessions")

storage.save("chat-2026-07-28", conv, extra={"model": "qwen2.5-7b"})

for s in storage.list(limit=10):
    print(s.session_id, s.message_count, s.preview)

loaded = storage.load("chat-2026-07-28")
conv2 = Conversation.from_dict({"messages": [m.__dict__ for m in loaded.messages]}, lg)
```

The `kelt session` CLI uses `FileSessionStorage` — same on-disk format.

## With `ContextQuery`

Pass a conversation to `ContextQuery.ask()` and the exchange is appended in place:

```python
conv = Conversation(lg, config=Config(max_tokens=8000))

async with ContextQuery(client=llm_client, context_builder=builder, base_system_prompt="…") as q:
    for question in [
        "What are Python generators?",
        "How do they differ from list comprehensions?",
        "When should I prefer one?",
    ]:
        answer = await q.ask(question, conversation=conv)
        print("Q:", question)
        print("A:", answer)
        storage.save("tutorial", conv)  # persist after each turn
```

`ContextQuery` writes both the user question and the assistant answer into `conv` — the next
call sees the full history and stays coherent.

## When a message won't fit

```python
try:
    conv.append(Message(role="user", content="…huge…"))
except ContextOverflowError as e:
    lg.warning(
        "message too big",
        extra={
            "token_count": e.token_count,
            "max_tokens": e.max_tokens,
        },
    )
    # fallback: summarise the input, drop it, or start a new session
```

Even with a compactor configured, this fires if the single new message alone exceeds
`max_tokens` minus the preserved head/tail. The fix is upstream: chunk the message, or raise
`max_tokens`.

## Full working script

See [`examples/conversation.py`](../examples/conversation.py) — covers messages, tool
calls, token accounting, sliding-window compaction, file storage (save/list/load/delete), and
optionally a multi-turn LLM conversation with per-turn persistence.
