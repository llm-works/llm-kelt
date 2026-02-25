"""
End-to-end test for fact memorization.

Tests that:
1. User provides a fact
2. Fact is stored
3. AI response differs with vs without facts

Requires local LLM (configured via LEARN_TEST_CONFIG_FILE).
"""

import pytest

from llm_kelt.client import Client
from llm_kelt.inference.context import ContextBuilder


@pytest.fixture
def facts_kelt_client(logger, database, test_context):
    """Create Client for facts testing."""
    from llm_kelt import ClientContext

    context = ClientContext(context_key=test_context, schema_name=None)
    return Client(database=database, context=context, lg=logger)


@pytest.mark.llm
class TestFactsEndToEnd:
    """End-to-end tests for fact memorization affecting LLM responses."""

    @pytest.mark.asyncio
    async def test_fact_affects_response(self, facts_kelt_client, llm_client, clean_tables):
        """
        Test that adding a fact about preferences changes the LLM response.

        This is the simplest possible test of fact memorization:
        - Query without facts → generic response
        - Add fact about language preference
        - Query with facts → response should reflect the preference
        """
        base_prompt = "You are a helpful assistant."
        question = "What programming language would you recommend for a new backend project?"

        # Step 1: Query WITHOUT facts
        response_without_facts = (
            await llm_client.chat_async(
                messages=[{"role": "user", "content": question}],
                system=base_prompt,
                temperature=0.3,  # Lower temperature for more consistent responses
            )
        ).content

        # Step 2: Add facts about preferences
        facts_kelt_client.atomic.assertions.add(
            "Preferred programming language: Python",
            category="preferences",
        )
        facts_kelt_client.atomic.assertions.add(
            "Prefers Python frameworks like FastAPI and Django",
            category="preferences",
        )

        # Step 3: Build system prompt WITH facts
        context_builder = ContextBuilder(facts_kelt_client.atomic.assertions)
        prompt_with_facts = context_builder.build_system_prompt(base_prompt)

        # Verify facts are in the prompt
        assert "Preferred programming language: Python" in prompt_with_facts
        assert "FastAPI" in prompt_with_facts

        # Step 4: Query WITH facts
        response_with_facts = (
            await llm_client.chat_async(
                messages=[{"role": "user", "content": question}],
                system=prompt_with_facts,
                temperature=0.3,
            )
        ).content

        # Step 5: Verify the responses are different
        # The response with facts should be more context-aware
        print("\n" + "=" * 60)
        print("RESPONSE WITHOUT FACTS:")
        print("=" * 60)
        print(response_without_facts)
        print("\n" + "=" * 60)
        print("RESPONSE WITH FACTS:")
        print("=" * 60)
        print(response_with_facts)
        print("\n" + "=" * 60)
        print("SYSTEM PROMPT WITH FACTS:")
        print("=" * 60)
        print(prompt_with_facts)
        print("=" * 60)

        # Basic assertion: responses should be different
        assert response_without_facts != response_with_facts, (
            "Responses should differ when facts are provided"
        )

        # The response with facts should mention Python given the preference
        response_lower = response_with_facts.lower()
        assert "python" in response_lower, (
            "Response with facts should mention Python given the stated preference"
        )

    @pytest.mark.asyncio
    async def test_fact_categories_affect_response(
        self, facts_kelt_client, llm_client, clean_tables
    ):
        """Test that different fact categories can be selectively included."""
        base_prompt = "You are a helpful assistant."
        question = "How should I approach learning a new technology?"

        # Add facts in different categories
        facts_kelt_client.atomic.assertions.add(
            "Learning style: hands-on projects",
            category="preferences",
        )
        facts_kelt_client.atomic.assertions.add(
            "Prefers short focused learning sessions",
            category="context",
        )
        facts_kelt_client.atomic.assertions.add(
            "Always suggest practical examples over theory",
            category="rules",
        )

        # Build prompt with only preferences
        context_builder = ContextBuilder(facts_kelt_client.atomic.assertions)
        prompt_prefs_only = context_builder.build_system_prompt(
            base_prompt,
            categories=["preferences"],
        )

        # Verify only preferences are included
        assert "hands-on projects" in prompt_prefs_only
        assert "short focused" not in prompt_prefs_only
        assert "practical examples" not in prompt_prefs_only

        # Build prompt with all facts
        prompt_all_facts = context_builder.build_system_prompt(base_prompt)

        # Verify all facts are included
        assert "hands-on projects" in prompt_all_facts
        assert "short focused" in prompt_all_facts
        assert "practical examples" in prompt_all_facts

        # Query with all facts
        response = (
            await llm_client.chat_async(
                messages=[{"role": "user", "content": question}],
                system=prompt_all_facts,
                temperature=0.3,
            )
        ).content

        print("\n" + "=" * 60)
        print("RESPONSE WITH ALL FACTS:")
        print("=" * 60)
        print(response)

        # Response should reflect learning preferences
        # Note: 0.5B models need some vocabulary flexibility, but avoid ultra-generic terms
        response_lower = response.lower()
        practice_terms = [
            # Core practice-oriented terms
            "hands-on",
            "practical",
            "project",
            "example",
            "build",
            "practice",
            "tutorial",
            "exercise",
            "apply",
            "code",
            "experiment",
        ]
        assert any(term in response_lower for term in practice_terms), (
            f"Response should reflect learning preferences. Got: {response[:200]}..."
        )

    @pytest.mark.asyncio
    async def test_deactivated_fact_not_used(self, facts_kelt_client, llm_client, clean_tables):
        """Test that deactivated facts are not included in prompts."""
        # Add and then deactivate a fact
        fact_id = facts_kelt_client.atomic.assertions.add(
            "Output format: XML only",
            category="preferences",
        )
        facts_kelt_client.atomic.assertions.deactivate(fact_id)

        # Add an active fact
        facts_kelt_client.atomic.assertions.add(
            "Output format: JSON preferred",
            category="preferences",
        )

        # Build prompt
        context_builder = ContextBuilder(facts_kelt_client.atomic.assertions)
        prompt = context_builder.build_system_prompt("You are a helpful assistant.")

        # Deactivated fact should not be in prompt
        assert "XML only" not in prompt
        assert "JSON preferred" in prompt
