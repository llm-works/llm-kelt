#!/usr/bin/env python3
"""Example: RAG (Retrieval-Augmented Generation) with Semantic Search.

This example demonstrates:
1. Embedding facts for semantic search
2. Similarity search with vector queries
3. RAG-based context injection vs static injection
4. How RAG selects relevant facts based on the question

Prerequisites:
    - PostgreSQL database with pgvector extension
    - Config file at etc/llm-learn.yaml
    - Embedding server running (e.g., llm-infer with embedding model)
    - LLM backend for chat (optional)

Usage:
    python examples/02_rag_retrieval.py
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
    ensure_demo_profile,
    psql_cmd,
)
from appinfra.config import Config
from appinfra.log import LogConfig, Logger, LoggerFactory
from httpx import ConnectError, ConnectTimeout
from llm_infer.client import Factory as LLMClientFactory

from llm_learn import LearnClient, LearnClientFactory
from llm_learn.inference import (
    ContextBuilder,
    ContextQuery,
    Embedder,
    RAGArgs,
    embed_missing_facts,
)

# Sample facts covering different domains for RAG demo
_SAMPLE_FACTS = [
    # Database-related
    ("PostgreSQL supports JSONB for efficient JSON storage and querying", "database"),
    ("Use connection pooling to avoid database connection overhead", "database"),
    ("Indexes should be created for frequently queried columns", "database"),
    # API-related
    ("REST APIs should use proper HTTP status codes (200, 400, 404, 500)", "api"),
    ("API rate limiting prevents abuse and ensures fair usage", "api"),
    ("Use OpenAPI/Swagger for API documentation", "api"),
    # Security-related
    ("Never store passwords in plain text, use bcrypt or argon2", "security"),
    ("JWT tokens should have short expiration times", "security"),
    ("Always validate and sanitize user input", "security"),
    # Testing-related
    ("Unit tests should be fast and isolated", "testing"),
    ("Integration tests verify component interactions", "testing"),
    ("Use mocking to isolate external dependencies in tests", "testing"),
]


def populate_facts(learn: LearnClient):
    """Clear and populate sample facts for the demo."""
    print(f"\n{H2}▶ Populating Facts{RESET}")

    existing = learn.facts.list_active()
    if existing:
        for fact in existing:
            learn.facts.deactivate(fact.id)
        print(f"  {MUTED}Cleared {len(existing)} existing facts{RESET}")

    for content, category in _SAMPLE_FACTS:
        learn.facts.add(content, category=category)

    print(f"  {OK}✓ Added {len(_SAMPLE_FACTS)} facts across 4 categories:{RESET}")
    categories: dict[str, list] = {}
    for fact in learn.facts.list_active():
        categories.setdefault(fact.category or "uncategorized", []).append(fact)
    for cat, facts in sorted(categories.items()):
        print(f"    {INFO}[{cat}]{RESET} {len(facts)} facts")

    print(
        f'\n  {CMD}▸ Verify:{RESET} {psql_cmd(learn)} -c "SELECT id, category, content FROM facts WHERE profile_id={learn.profile_id} AND active=true;"'
    )


async def embed_facts(lg: Logger, learn: LearnClient, config: Config) -> Embedder | None:
    """Embed facts for semantic search. Returns embedder if successful."""
    print(f"\n{H2}▶ Embedding Facts for Semantic Search{RESET}")

    embedding_config = config.embedding
    embedder = Embedder(base_url=embedding_config.base_url, model=embedding_config.model_name)

    try:
        print(f"  {MUTED}Connecting to embedding server...{RESET}")
        result = await embed_missing_facts(
            lg=lg, embedder=embedder, embedding_adapter=learn.embeddings, batch_size=50
        )
        print(f"  {OK}✓ Embedded {result.processed} facts{RESET}")
        if result.failed:
            print(f"  {WARN}⚠ {result.failed} failed{RESET}")

        print(
            f'\n  {CMD}▸ Verify embeddings:{RESET} {psql_cmd(learn)} -c "SELECT id, content, (embedding IS NOT NULL) as has_embedding FROM facts WHERE profile_id={learn.profile_id} LIMIT 5;"'
        )
        return embedder
    except (ConnectError, ConnectTimeout, OSError):
        # Server not running - fall back to synthetic embeddings
        pass

    print(f"  {MUTED}[Skipped] Embedding server not available{RESET}")
    print(f"  {MUTED}Using synthetic embeddings for demo...{RESET}")

    # Set demo embeddings manually based on keywords
    for i, fact in enumerate(learn.facts.list_active()):
        embedding = [0.0] * 384
        content_lower = fact.content.lower()
        # Create simple keyword-based embeddings for demo
        if "database" in content_lower or "postgres" in content_lower or "sql" in content_lower:
            embedding[0] = 0.9
        if "api" in content_lower or "rest" in content_lower or "http" in content_lower:
            embedding[1] = 0.9
        if "security" in content_lower or "password" in content_lower or "jwt" in content_lower:
            embedding[2] = 0.9
        if "test" in content_lower or "mock" in content_lower:
            embedding[3] = 0.9
        embedding[i % 384] = 0.5  # Ensure uniqueness
        learn.embeddings.set_embedding(fact.id, embedding, "demo-model")

    print(f"  {OK}✓ Set synthetic embeddings for all facts{RESET}")
    return None


async def demo_similarity_search(learn: LearnClient, embedder: Embedder):
    """Demonstrate similarity search with embeddings."""
    print(f"\n{H2}▶ Similarity Search{RESET}")
    print(f"  {MUTED}Finding facts similar to a query using vector similarity{RESET}")

    query = "How do I secure my database connections?"
    print(f'\n  {LLM_Q}Query: "{query}"{RESET}')

    query_result = await embedder.embed_async(query)
    similar_facts = learn.embeddings.search_similar(
        query=query_result.embedding, model_name=embedder.model, top_k=5, min_similarity=0.3
    )

    print(f"  {OK}Top matches:{RESET}")
    for sf in similar_facts:
        print(
            f"    {MUTED}[{sf.score:.3f}]{RESET} {INFO}[{sf.entity.category}]{RESET} {sf.entity.content}"
        )

    # Search with category filter
    print(f"\n  {LLM_Q}Query: \"{query}\" {MUTED}(filtered to 'security' category){RESET}")

    similar_security = learn.embeddings.search_similar(
        query=query_result.embedding,
        model_name=embedder.model,
        top_k=5,
        min_similarity=0.3,
        categories=["security"],
    )

    print(f"  {OK}Security-only matches:{RESET}")
    for sf in similar_security:
        print(
            f"    {MUTED}[{sf.score:.3f}]{RESET} {INFO}[{sf.entity.category}]{RESET} {sf.entity.content}"
        )


async def demo_rag_vs_static(learn: LearnClient, config: Config, embedder: Embedder | None):
    """Demonstrate RAG vs static context injection - the key comparison."""
    print(f"\n{H2}▶ RAG vs Static Context Injection{RESET}")
    print(f"  {MUTED}Comparing which facts get included in the LLM prompt{RESET}")

    context_builder = ContextBuilder(learn.facts)
    question = "How should I handle database queries securely?"

    # Static: just takes first N facts regardless of relevance
    print(f"\n  {WARN}Static injection{RESET} (first 3 facts, ignores question):")
    static_prompt = context_builder.build_system_prompt(
        base_prompt="You are a coding assistant.", max_facts=3
    )
    # Extract just the facts from the prompt
    lines = static_prompt.split("\n")
    fact_lines = [ln for ln in lines if ln.startswith("- ")]
    for line in fact_lines[:3]:
        print(f"    {MUTED}•{RESET} {line[2:]}")

    # RAG: selects facts based on semantic similarity to the question
    print(f"\n  {OK}RAG injection{RESET} (facts selected by similarity to question):")
    print(f'  {LLM_Q}Question: "{question}"{RESET}')

    if embedder:
        query_result = await embedder.embed_async(question)
        similar = learn.embeddings.search_similar(
            query=query_result.embedding, model_name=embedder.model, top_k=3, min_similarity=0.3
        )
        print(f"  {OK}Selected facts:{RESET}")
        for sf in similar:
            print(
                f"    {MUTED}[{sf.score:.3f}]{RESET} {INFO}[{sf.entity.category}]{RESET} {sf.entity.content}"
            )
    else:
        print(f"    {MUTED}(Requires embedding server for live demo){RESET}")

    print(
        f"\n  {INFO}ℹ RAG selects 'database' and 'security' facts because they're relevant to the question{RESET}"
    )


async def demo_rag_query(learn: LearnClient, config: Config, lg: Logger, embedder: Embedder | None):
    """Demonstrate full RAG query with LLM."""
    print(f"\n{H2}▶ Full RAG Query (LLM + Context){RESET}")

    try:
        llm_factory = LLMClientFactory(lg)
        llm_client = llm_factory.from_config(config.llm.to_dict())
        context_builder = ContextBuilder(learn.facts)

        query = ContextQuery(
            client=llm_client,
            context_builder=context_builder,
            base_system_prompt="You are a coding assistant.",
            embedder=embedder,
            embedding_adapter=learn.embeddings,
        )

        question = "What are best practices for API security?"
        print(f"\n  {LLM_Q}Q: {question}{RESET}")
        print(f"  {MUTED}(RAG finds relevant security and API facts...){RESET}")

        response = await query.ask(question, rag=RAGArgs(top_k=4, min_similarity=0.3))
        response_clean = response.replace("<think>\n\n</think>\n\n", "").strip()
        print(f"  {LLM_A}{response_clean[:500]}{RESET}")
        if len(response_clean) > 500:
            print(f"  {MUTED}[...truncated]{RESET}")

        await llm_client.aclose()

    except Exception as e:
        print(f"  {MUTED}[Skipped] No LLM backend: {type(e).__name__}{RESET}")
        print(
            f"  {MUTED}Start llm-infer or configure OpenAI in etc/llm-learn.yaml to enable.{RESET}"
        )


async def main():
    """Run the RAG retrieval demo."""
    print(f"\n{H1}{'━' * 50}{RESET}")
    print(f"{H1}  Example 02: RAG (Retrieval-Augmented Generation){RESET}")
    print(f"{H1}{'━' * 50}{RESET}")

    # Suppress logging noise
    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
    config = Config("etc/llm-learn.yaml")
    factory = LearnClientFactory(lg)

    # Create initial client to get/create profile
    learn = factory.create_from_config(profile_id="1", config=config)
    profile_id = ensure_demo_profile(learn)
    learn = factory.create_from_config(profile_id=profile_id, config=config)
    print(f"{MUTED}Using profile_id={RESET}{INFO}{profile_id}{RESET}")

    # Run demos
    populate_facts(learn)
    embedder = await embed_facts(lg, learn, config)

    if embedder:
        await demo_similarity_search(learn, embedder)

    await demo_rag_vs_static(learn, config, embedder)
    await demo_rag_query(learn, config, lg, embedder)

    if embedder:
        await embedder.close_async()

    print(f"\n{H1}{'━' * 50}{RESET}")
    print(f"{OK}✓ Done!{RESET} Next: {CMD}python examples/03_training_export.py{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
