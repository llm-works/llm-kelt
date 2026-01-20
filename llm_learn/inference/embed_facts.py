"""Utilities for batch embedding facts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from appinfra.log import Logger

from ..collection.facts import FactsClient
from ..core.models import Fact
from .embedder import Embedder, EmbeddingResult

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class EmbedFactsResult:
    """Result from batch embedding operation."""

    processed: int
    failed: int


def _store_embeddings(
    facts: "Sequence[Fact]",
    results: list[EmbeddingResult],
    facts_client: FactsClient,
    model_name: str,
    lg: Logger,
) -> tuple[int, int]:
    """Store embeddings for facts, returning (processed, failed) counts."""
    processed = 0
    failed = 0
    for fact, result in zip(facts, results):
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
    facts: "Sequence[Fact]",
    embedder: Embedder,
    facts_client: FactsClient,
    model_name: str,
    lg: Logger,
) -> tuple[int, int]:
    """Embed facts one at a time as fallback, returning (processed, failed) counts."""
    processed = 0
    failed = 0
    for fact in facts:
        try:
            result = await embedder.embed(fact.content)
            facts_client.set_embedding(fact.id, result.embedding, model_name)
            processed += 1
        except Exception as e:
            lg.error(
                "failed to embed fact",
                extra={"fact_id": fact.id, "exception": e},
            )
            failed += 1
    return processed, failed


async def embed_missing_facts(
    lg: Logger,
    embedder: Embedder,
    facts_client: FactsClient,
    model_name: str,
    batch_size: int = 50,
) -> EmbedFactsResult:
    """
    Embed all facts that don't have embeddings for the specified model.

    Finds facts missing embeddings and generates them in batches.
    Continues processing even if individual embeddings fail.

    Args:
        lg: Logger instance.
        embedder: Embedder client for generating embeddings.
        facts_client: FactsClient for retrieving and updating facts.
        model_name: Name of the embedding model (stored with embeddings).
        batch_size: Number of facts to embed per batch.

    Returns:
        EmbedFactsResult with counts of processed and failed facts.
    """
    processed = 0
    failed = 0

    while True:
        facts = facts_client.list_without_embeddings(model_name, limit=batch_size)
        if not facts:
            break

        try:
            results = await embedder.embed_batch([f.content for f in facts])
            p, f = _store_embeddings(facts, results, facts_client, model_name, lg)
        except Exception as e:
            lg.warning(
                "batch embedding failed, falling back to individual",
                extra={"batch_size": len(facts), "exception": e},
            )
            p, f = await _embed_individually(facts, embedder, facts_client, model_name, lg)

        processed += p
        failed += f

    return EmbedFactsResult(processed=processed, failed=failed)
