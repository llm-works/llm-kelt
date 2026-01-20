"""Integration tests for RAG functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_learn.inference.context import ContextBuilder
from llm_learn.inference.embedder import EmbeddingResult
from llm_learn.inference.query import ContextQuery, RAGArgs


class TestRAGIntegration:
    """Integration tests for RAG with real database."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock Embedder that returns deterministic embeddings based on content."""
        embedder = MagicMock()

        # Embedding vectors that are similar for related concepts
        embeddings_map = {
            # Python-related
            "python": [0.9, 0.1, 0.0],
            "backend": [0.8, 0.2, 0.0],
            "api": [0.7, 0.3, 0.0],
            # Preferences-related
            "concise": [0.1, 0.9, 0.0],
            "brief": [0.1, 0.8, 0.1],
            "short": [0.1, 0.7, 0.2],
            # Unrelated
            "music": [0.0, 0.0, 1.0],
            "cooking": [0.0, 0.1, 0.9],
        }

        def get_embedding(text: str) -> list[float]:
            """Get embedding based on keywords in text."""
            text_lower = text.lower()
            for keyword, emb in embeddings_map.items():
                if keyword in text_lower:
                    return emb
            # Default embedding
            return [0.3, 0.3, 0.3]

        async def embed(text):
            return EmbeddingResult(
                embedding=get_embedding(text),
                model="test-model",
                prompt_tokens=len(text),
            )

        embedder.embed = AsyncMock(side_effect=embed)
        embedder._model = "test-model"
        return embedder

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.chat = AsyncMock(return_value="Test response from LLM")
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def sample_facts_with_embeddings(self, learn_client, clean_tables):
        """Create facts with embeddings for similarity search."""
        facts_data = [
            # Python/backend related facts
            ("User prefers Python for backend development", "preferences", [0.85, 0.15, 0.0]),
            ("User has experience with REST API design", "background", [0.75, 0.25, 0.0]),
            # Preference for concise explanations
            ("User likes concise explanations", "preferences", [0.1, 0.85, 0.05]),
            ("Keep responses brief and to the point", "rules", [0.1, 0.75, 0.15]),
            # Unrelated facts
            ("User enjoys classical music", "background", [0.0, 0.05, 0.95]),
            ("Hobby: cooking Italian food", "background", [0.0, 0.1, 0.9]),
        ]

        fact_ids = []
        for content, category, embedding in facts_data:
            fact_id = learn_client.facts.add(content, category=category)
            learn_client.facts.set_embedding(fact_id, embedding, "test-model")
            fact_ids.append(fact_id)

        return fact_ids

    @pytest.mark.asyncio
    async def test_rag_retrieves_relevant_facts(
        self, learn_client, mock_embedder, mock_llm_client, sample_facts_with_embeddings
    ):
        """Test that RAG retrieves facts relevant to the query."""
        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        # Ask about Python - should retrieve Python-related facts
        await query.ask(
            "What language should I use for my API backend?",
            rag=RAGArgs(top_k=5, min_similarity=0.5),
        )

        # Check what was passed to the LLM
        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"]

        # Should include Python-related facts
        assert "Python for backend" in system_prompt
        assert "REST API" in system_prompt

        # Should NOT include unrelated facts (low similarity)
        assert "classical music" not in system_prompt
        assert "cooking" not in system_prompt

    @pytest.mark.asyncio
    async def test_rag_excludes_unrelated_facts(
        self, learn_client, mock_embedder, mock_llm_client, sample_facts_with_embeddings
    ):
        """Test that RAG excludes facts below similarity threshold."""
        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        # Ask about concise responses - "brief" keyword triggers [0.1, 0.8, 0.1] embedding
        await query.ask(
            "I want brief explanations",  # Will get "brief" embedding
            rag=RAGArgs(top_k=10, min_similarity=0.8),  # High threshold
        )

        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"] or ""

        # With high similarity threshold, should get facts with similar embeddings
        # The "brief" embedding [0.1, 0.8, 0.1] is similar to conciseness facts
        # and dissimilar to Python facts [0.9, 0.1, 0.0] and music facts [0.0, 0.0, 1.0]

        # Either no facts match (empty prompt) or only concise-related facts match
        # This test verifies the similarity filter is working
        if "About the user" in system_prompt:
            # If facts were found, they should be related to conciseness, not Python/music
            assert "classical music" not in system_prompt
            assert "cooking" not in system_prompt

    @pytest.mark.asyncio
    async def test_rag_with_no_embedded_facts(
        self, learn_client, mock_embedder, mock_llm_client, clean_tables
    ):
        """Test RAG gracefully handles when no facts have embeddings."""
        # Add facts but don't embed them
        learn_client.facts.add("User prefers TypeScript")
        learn_client.facts.add("Output format: JSON")

        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        # Should work but return no facts
        await query.ask(
            "What's my preferred language?",
            rag=RAGArgs(top_k=5, min_similarity=0.3),
        )

        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"] or ""

        # No facts should be included (they have no embeddings)
        assert "TypeScript" not in system_prompt
        assert "JSON" not in system_prompt

    @pytest.mark.asyncio
    async def test_rag_respects_top_k(
        self, learn_client, mock_embedder, mock_llm_client, sample_facts_with_embeddings
    ):
        """Test that RAG respects top_k limit."""
        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        # Set very low min_similarity to get all facts, but limit to 2
        await query.ask(
            "Tell me about Python",
            rag=RAGArgs(top_k=2, min_similarity=0.0),
        )

        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"]

        # Count how many facts are in the prompt (each starts with "- ")
        fact_count = system_prompt.count("\n- ")
        # Should have at most 2 facts (top_k=2)
        assert fact_count <= 2

    @pytest.mark.asyncio
    async def test_rag_respects_min_similarity(
        self, learn_client, mock_embedder, mock_llm_client, sample_facts_with_embeddings
    ):
        """Test that RAG respects min_similarity threshold."""
        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        # Set very high min_similarity - should return very few or no facts
        await query.ask(
            "Random query",  # Will get default embedding [0.3, 0.3, 0.3]
            rag=RAGArgs(top_k=10, min_similarity=0.99),
        )

        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"] or ""

        # With 0.99 similarity threshold, probably no facts match
        assert "About the user" not in system_prompt or system_prompt.count("- ") == 0

    @pytest.mark.asyncio
    async def test_rag_only_searches_active_facts(
        self, learn_client, mock_embedder, mock_llm_client, clean_tables
    ):
        """Test that RAG only retrieves active facts."""
        # Create active and inactive facts with same embedding
        active_id = learn_client.facts.add("Active fact about Python")
        inactive_id = learn_client.facts.add("Inactive fact about Python")

        # Give both the same Python-related embedding
        learn_client.facts.set_embedding(active_id, [0.9, 0.1, 0.0], "test-model")
        learn_client.facts.set_embedding(inactive_id, [0.9, 0.1, 0.0], "test-model")

        # Deactivate one
        learn_client.facts.deactivate(inactive_id)

        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        await query.ask(
            "Tell me about Python",
            rag=RAGArgs(top_k=10, min_similarity=0.3),
        )

        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"]

        # Only active fact should be included
        assert "Active fact" in system_prompt
        assert "Inactive fact" not in system_prompt

    @pytest.mark.asyncio
    async def test_rag_with_custom_system_prompt(
        self, learn_client, mock_embedder, mock_llm_client, sample_facts_with_embeddings
    ):
        """Test RAG works with custom system prompt."""
        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            base_system_prompt="You are a helpful coding assistant.",
            embedder=mock_embedder,
        )

        await query.ask(
            "What's my preferred language?",
            rag=RAGArgs(top_k=5, min_similarity=0.5),
        )

        call_kwargs = mock_llm_client.chat.call_args.kwargs
        system_prompt = call_kwargs["system"]

        # Should include both base prompt and facts
        assert "helpful coding assistant" in system_prompt
        assert "About the user:" in system_prompt

    @pytest.mark.asyncio
    async def test_rag_multi_turn_conversation(
        self, learn_client, mock_embedder, mock_llm_client, sample_facts_with_embeddings
    ):
        """Test RAG works across multi-turn conversations."""
        from llm_learn.inference.query import Conversation

        context_builder = ContextBuilder(learn_client.facts)
        query = ContextQuery(
            client=mock_llm_client,
            context_builder=context_builder,
            embedder=mock_embedder,
        )

        conv = Conversation()

        # First turn - Python question
        await query.ask(
            "What language should I use?",
            conversation=conv,
            rag=RAGArgs(top_k=5, min_similarity=0.5),
        )

        # Second turn - Follow up (will embed this new question)
        await query.ask(
            "Tell me more about that",
            conversation=conv,
            rag=RAGArgs(top_k=5, min_similarity=0.0),  # Low threshold for follow-up
        )

        # Conversation should have all messages
        assert len(conv.messages) == 4  # 2 user + 2 assistant
        assert conv.messages[0].content == "What language should I use?"
        assert conv.messages[2].content == "Tell me more about that"

        # Embedder should have been called twice (once per question)
        assert mock_embedder.embed.call_count == 2


class TestBuildSystemPromptFromFacts:
    """Test ContextBuilder.build_system_prompt_from_facts method."""

    def test_builds_prompt_with_facts(self, learn_client, clean_tables):
        """Test building prompt from provided facts."""
        # Create and fetch facts
        learn_client.facts.add("Fact one", category="prefs")
        learn_client.facts.add("Fact two", category="prefs")
        facts = learn_client.facts.list_active()

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt_from_facts(
            base_prompt="You are an assistant.",
            facts=facts,
        )

        assert "You are an assistant." in result
        assert "About the user:" in result
        assert "Fact one" in result
        assert "Fact two" in result

    def test_builds_prompt_with_empty_facts(self, learn_client, clean_tables):
        """Test building prompt with no facts returns base prompt."""
        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt_from_facts(
            base_prompt="Base prompt only",
            facts=[],
        )

        assert result == "Base prompt only"

    def test_builds_prompt_prepend(self, learn_client, clean_tables):
        """Test prepending facts to prompt."""
        learn_client.facts.add("Test fact")
        facts = learn_client.facts.list_active()

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt_from_facts(
            base_prompt="Base prompt",
            facts=facts,
            fact_position="prepend",
        )

        # Facts should come before base prompt
        facts_pos = result.find("About the user:")
        base_pos = result.find("Base prompt")
        assert facts_pos < base_pos

    def test_groups_facts_by_category(self, learn_client, clean_tables):
        """Test facts are grouped by category."""
        learn_client.facts.add("Pref 1", category="preferences")
        learn_client.facts.add("Pref 2", category="preferences")
        learn_client.facts.add("Rule 1", category="rules")
        facts = learn_client.facts.list_active()

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt_from_facts("", facts)

        assert "### Preferences" in result
        assert "### Rules" in result
