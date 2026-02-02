"""Utilities for batch embedding facts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from appinfra.log import Logger

from ..memory.v1.clients import AssertionsClient
from ..memory.v1.models import Fact
from .embedder import Embedder, EmbeddingResult

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class EmbedFactsResult:
    """Result from batch embedding operation."""

    processed: int
    failed: int


def _store_embeddings(
    lg: Logger,
    facts: "Sequence[Fact]",
    results: list[EmbeddingResult],
    facts_client: AssertionsClient,
    model_name: str,
) -> tuple[int, int]:
    """Store embeddings for facts, returning (processed, failed) counts."""
    processed = 0
    failed = 0
    for fact, result in zip(facts, results, strict=True):
        try:
            facts_client.set_embedding(fact.id, result.embedding, model_name)
            processed += 1
        except Exception as e:
            lg.error(
                "failed to store embedding",
                extra={"fact_id": fact.id, "exception": e},
            )
            failed += 1
    return processed, failed


async def _embed_individually(
    lg: Logger,
    facts: "Sequence[Fact]",
    embedder: Embedder,
    facts_client: AssertionsClient,
    model_name: str,
) -> tuple[int, int]:
    """Embed facts one at a time as fallback, returning (processed, failed) counts."""
    processed = 0
    failed = 0
    for fact in facts:
        try:
            result = await embedder.embed_async(fact.content)
            facts_client.set_embedding(fact.id, result.embedding, model_name)
            processed += 1
        except Exception as e:
            lg.error(
                "failed to embed fact",
                extra={"fact_id": fact.id, "exception": e},
            )
            failed += 1
    return processed, failed


async def _process_batch(
    lg: Logger,
    facts: "Sequence[Fact]",
    embedder: Embedder,
    facts_client: AssertionsClient,
    model_name: str,
) -> tuple[int, int]:
    """Process a single batch of facts, with fallback to individual embedding."""
    try:
        results = await embedder.embed_batch_async([f.content for f in facts])
        return _store_embeddings(lg, facts, results, facts_client, model_name)
    except Exception as e:
        lg.warning(
            "batch embedding failed, falling back to individual",
            extra={"batch_size": len(facts), "exception": e},
        )
        return await _embed_individually(lg, facts, embedder, facts_client, model_name)


async def embed_missing_facts(
    lg: Logger,
    embedder: Embedder,
    facts_client: AssertionsClient,
    batch_size: int = 50,
) -> EmbedFactsResult:
    """
    Embed all facts that don't have embeddings for the embedder's model.

    Finds facts missing embeddings and generates them in batches.
    Continues processing even if individual embeddings fail.

    The model name is discovered from the embedding server automatically.

    Args:
        lg: Logger instance.
        embedder: Embedder client for generating embeddings.
        facts_client: AssertionsClient for retrieving and updating facts.
        batch_size: Number of facts to embed per batch.

    Returns:
        EmbedFactsResult with counts of processed and failed facts.
    """
    model_name = await embedder.discover_async()
    processed = 0
    failed = 0

    while True:
        facts = facts_client.list_without_embeddings(model_name, limit=batch_size)
        if not facts:
            break

        p, f = await _process_batch(lg, facts, embedder, facts_client, model_name)
        processed += p
        failed += f

        if p == 0:
            lg.warning(
                "no progress made in batch, stopping to avoid infinite loop",
                extra={"failed_in_batch": f},
            )
            break

    return EmbedFactsResult(processed=processed, failed=failed)
