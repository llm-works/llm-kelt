"""Utilities for batch embedding facts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from appinfra.log import Logger
from llm_infer.client import EmbeddingClient

from ..memory.atomic.embedding import EmbeddingAdapter
from ..memory.atomic.models import Fact

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
    embeddings: list[list[float]],
    embedding_adapter: EmbeddingAdapter,
    model: str,
) -> tuple[int, int]:
    """Store embeddings for facts, returning (processed, failed) counts."""
    processed = 0
    failed = 0
    for fact, embedding in zip(facts, embeddings, strict=True):
        try:
            embedding_adapter.set_embedding(fact.id, embedding, model)
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
    embedder: EmbeddingClient,
    embedding_adapter: EmbeddingAdapter,
    model: str,
) -> tuple[int, int]:
    """Embed facts one at a time as fallback, returning (processed, failed) counts."""
    processed = 0
    failed = 0
    for fact in facts:
        try:
            result = await embedder.embed_async(fact.content)
            embedding_adapter.set_embedding(fact.id, result.embedding, model)
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
    embedder: EmbeddingClient,
    embedding_adapter: EmbeddingAdapter,
    model: str,
) -> tuple[int, int]:
    """Process a single batch of facts, with fallback to individual embedding."""
    try:
        batch_result = await embedder.embed_batch_async([f.content for f in facts])
        return _store_embeddings(lg, facts, batch_result.embeddings, embedding_adapter, model)
    except Exception as e:
        lg.warning(
            "batch embedding failed, falling back to individual",
            extra={"batch_size": len(facts), "exception": e},
        )
        return await _embed_individually(lg, facts, embedder, embedding_adapter, model)


async def embed_missing_facts(
    lg: Logger,
    embedder: EmbeddingClient,
    embedding_adapter: EmbeddingAdapter,
    dimensions: int,
    batch_size: int = 50,
) -> EmbedFactsResult:
    """
    Embed all facts that don't have embeddings for the embedder's model.

    Finds facts missing embeddings and generates them in batches.
    Continues processing even if individual embeddings fail.

    Uses the model name from the embedder.

    Args:
        lg: Logger instance.
        embedder: EmbeddingClient client for generating embeddings.
        embedding_adapter: EmbeddingAdapter for storing embeddings.
        dimensions: Output dimensions for embeddings (determines storage table).
        batch_size: Number of facts to embed per batch.

    Returns:
        EmbedFactsResult with counts of processed and failed facts.
    """
    model = embedder.model
    processed = 0
    failed = 0

    while True:
        facts = embedding_adapter.list_without_embeddings(model, dimensions, limit=batch_size)
        if not facts:
            break

        p, f = await _process_batch(lg, facts, embedder, embedding_adapter, model)
        processed += p
        failed += f

        if p == 0:
            lg.warning(
                "no progress made in batch, stopping to avoid infinite loop",
                extra={"failed_in_batch": f},
            )
            break

    return EmbedFactsResult(processed=processed, failed=failed)
