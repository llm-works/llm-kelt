"""Context builder for injecting facts into system prompts."""

from typing import Literal

from ..collection.facts import FactsClient
from ..core.models import Fact


class ContextBuilder:
    """
    Builds system prompts with injected user facts.

    Takes facts from the database and formats them into a section
    that gets prepended/appended to the base system prompt.

    Usage:
        builder = ContextBuilder(facts_client)
        system_prompt = builder.build_system_prompt(
            base_prompt="You are a helpful assistant.",
        )
        # Returns: "You are a helpful assistant.\n\n## About the user:\n- Fact 1\n- Fact 2"
    """

    def __init__(self, facts_client: FactsClient):
        """
        Initialize context builder.

        Args:
            facts_client: Client for retrieving facts from database
        """
        self._facts_client = facts_client

    def build_system_prompt(  # cq: max-lines=45
        self,
        base_prompt: str = "",
        categories: list[str] | None = None,
        min_confidence: float = 0.0,
        max_facts: int = 100,
        fact_position: str = "append",
    ) -> str:
        """
        Build a system prompt with injected facts.

        Args:
            base_prompt: The base system prompt to augment
            categories: Only include facts from these categories (None = all)
            min_confidence: Minimum confidence threshold for facts
            max_facts: Maximum number of facts to include
            fact_position: Where to place facts - "append" or "prepend"

        Returns:
            System prompt with facts section added
        """
        # Retrieve facts based on filters
        if categories:
            facts = []
            for category in categories:
                category_facts = self._facts_client.list_active(
                    category=category,
                    min_confidence=min_confidence,
                    limit=max_facts,
                )
                facts.extend(category_facts)
            # Dedupe and limit
            seen_ids = set()
            unique_facts = []
            for f in facts:
                if f.id not in seen_ids:
                    seen_ids.add(f.id)
                    unique_facts.append(f)
            facts = unique_facts[:max_facts]
        else:
            facts = self._facts_client.list_active(
                min_confidence=min_confidence,
                limit=max_facts,
            )

        # Format facts into a section
        facts_section = self._format_facts(facts)

        # Combine with base prompt
        if not facts_section:
            return base_prompt

        return self._combine_prompt_and_facts(base_prompt, facts_section, fact_position)

    def build_system_prompt_from_facts(
        self,
        base_prompt: str,
        facts: list[Fact],
        fact_position: Literal["append", "prepend"] = "append",
    ) -> str:
        """
        Build system prompt from pre-retrieved facts (for RAG).

        Unlike build_system_prompt which fetches facts from the database,
        this method accepts facts that were already retrieved (e.g., via
        semantic search).

        Args:
            base_prompt: The base system prompt to augment
            facts: List of facts to include (already retrieved)
            fact_position: Where to place facts - "append" or "prepend"

        Returns:
            System prompt with facts section added
        """
        facts_section = self._format_facts(facts)
        return self._combine_prompt_and_facts(base_prompt, facts_section, fact_position)

    def _combine_prompt_and_facts(
        self,
        base_prompt: str,
        facts_section: str,
        fact_position: str,
    ) -> str:
        """Combine base prompt with facts section."""
        if not facts_section:
            return base_prompt

        if fact_position == "prepend":
            if base_prompt:
                return f"{facts_section}\n\n{base_prompt}"
            return facts_section
        else:  # append
            if base_prompt:
                return f"{base_prompt}\n\n{facts_section}"
            return facts_section

    def _format_facts(self, facts: list[Fact]) -> str:
        """
        Format facts into a readable section.

        Args:
            facts: List of facts to format

        Returns:
            Formatted facts section (empty string if no facts)
        """
        if not facts:
            return ""

        lines = ["## About the user:"]

        # Group by category if facts have categories
        categorized: dict[str | None, list[Fact]] = {}
        for fact in facts:
            cat = fact.category
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(fact)

        # If all facts are uncategorized, just list them
        if len(categorized) == 1 and None in categorized:
            for fact in facts:
                lines.append(f"- {fact.content}")
        else:
            # Group by category
            for category, cat_facts in sorted(
                categorized.items(), key=lambda x: (x[0] is None, x[0] or "")
            ):
                if category:
                    lines.append(f"\n### {category.title()}")
                for fact in cat_facts:
                    lines.append(f"- {fact.content}")

        return "\n".join(lines)

    def get_facts_summary(self, max_facts: int = 100) -> dict:
        """
        Get a summary of current facts.

        Useful for debugging and inspection.

        Args:
            max_facts: Maximum facts to include

        Returns:
            Dict with counts, categories, and fact list
        """
        facts = self._facts_client.list_active(limit=max_facts)
        counts = self._facts_client.count_by_category()

        return {
            "total_active": len(facts),
            "by_category": counts,
            "facts": [
                {
                    "id": f.id,
                    "content": f.content,
                    "category": f.category,
                    "source": f.source,
                    "confidence": f.confidence,
                }
                for f in facts
            ],
        }
