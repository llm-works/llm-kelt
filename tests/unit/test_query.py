"""Unit tests for ContextQuery and RAG functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_learn.inference.embedder import EmbeddingResult
from llm_learn.inference.query import ContextQuery, Conversation, RAGArgs
from llm_learn.memory.v1 import Fact, ScoredFact


class TestRAGArgs:
    """Test RAGArgs dataclass."""

    def test_default_values(self):
        """Test RAGArgs default values."""
        args = RAGArgs()
        assert args.top_k == 10
        assert args.min_similarity == 0.3
        assert args.model_name is None
        assert args.categories is None

    def test_custom_values(self):
        """Test RAGArgs with custom values."""
        args = RAGArgs(
            top_k=5,
            min_similarity=0.5,
            model_name="custom-model",
            categories=["preferences", "rules"],
        )
        assert args.top_k == 5
        assert args.min_similarity == 0.5
        assert args.model_name == "custom-model"
        assert args.categories == ["preferences", "rules"]


class TestConversation:
    """Test Conversation class."""

    def test_add_user_message(self):
        """Test adding user message."""
        conv = Conversation()
        conv.add_user("Hello")
        assert len(conv.messages) == 1
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        """Test adding assistant message."""
        conv = Conversation()
        conv.add_assistant("Hi there")
        assert len(conv.messages) == 1
        assert conv.messages[0]["role"] == "assistant"
        assert conv.messages[0]["content"] == "Hi there"

    def test_clear(self):
        """Test clearing conversation."""
        conv = Conversation()
        conv.add_user("Hello")
        conv.add_assistant("Hi")
        conv.clear()
        assert len(conv.messages) == 0


class TestContextQueryRAG:
    """Test ContextQuery RAG functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.chat_async = AsyncMock(return_value="Test response")
        client.aclose = AsyncMock()
        return client

    @pytest.fixture
    def mock_facts_client(self):
        """Create mock facts client."""
        return MagicMock()

    @pytest.fixture
    def mock_context_builder(self, mock_facts_client):
        """Create mock context builder."""
        builder = MagicMock()
        builder.facts_client = mock_facts_client
        builder.build_system_prompt.return_value = "System prompt with facts"
        builder.build_system_prompt_from_facts.return_value = "RAG system prompt"
        return builder

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        embedder.embed_async = AsyncMock(
            return_value=EmbeddingResult(
                embedding=[0.1, 0.2, 0.3],
                model="test-model",
                prompt_tokens=5,
            )
        )
        embedder.model = "test-model"
        return embedder

    @pytest.fixture
    def sample_fact(self):
        """Create a sample fact."""
        return Fact(
            id=1,
            profile_id=1,
            content="User prefers Python",
            category="preferences",
            source="user",
            confidence=1.0,
            active=True,
        )

    @pytest.mark.asyncio
    async def test_ask_with_rag_embeds_question(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG embeds the question."""
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=sample_fact, similarity=0.9)
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("What language should I use?", rag=RAGArgs())

        mock_embedder.embed_async.assert_called_once_with("What language should I use?")

    @pytest.mark.asyncio
    async def test_ask_with_rag_searches_similar_facts(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG searches for similar facts with correct params."""
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=sample_fact, similarity=0.9)
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("Test question", rag=RAGArgs(top_k=5, min_similarity=0.4))

        mock_context_builder.facts_client.search_similar.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
            top_k=5,
            min_similarity=0.4,
            categories=None,
        )

    @pytest.mark.asyncio
    async def test_ask_with_rag_injects_retrieved_facts(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG injects retrieved facts into system prompt."""
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=sample_fact, similarity=0.9)
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            base_system_prompt="You are helpful.",
            embedder=mock_embedder,
        )

        await query.ask("Test", rag=RAGArgs())

        mock_context_builder.build_system_prompt_from_facts.assert_called_once_with(
            "You are helpful.", [sample_fact]
        )
        mock_client.chat_async.assert_called_once()
        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["system"] == "RAG system prompt"

    @pytest.mark.asyncio
    async def test_ask_with_rag_no_embedder_raises(self, mock_client, mock_context_builder):
        """Test that RAG without embedder raises ValueError."""
        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
        )

        with pytest.raises(ValueError, match="RAG requires embedder"):
            await query.ask("Test", rag=RAGArgs())

    @pytest.mark.asyncio
    async def test_ask_with_rag_config_top_k(
        self, mock_client, mock_context_builder, mock_embedder
    ):
        """Test that RAGArgs.top_k is passed to search."""
        mock_context_builder.facts_client.search_similar.return_value = []

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("Test", rag=RAGArgs(top_k=20))

        call_kwargs = mock_context_builder.facts_client.search_similar.call_args.kwargs
        assert call_kwargs["top_k"] == 20

    @pytest.mark.asyncio
    async def test_ask_with_rag_config_min_similarity(
        self, mock_client, mock_context_builder, mock_embedder
    ):
        """Test that RAGArgs.min_similarity is passed to search."""
        mock_context_builder.facts_client.search_similar.return_value = []

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("Test", rag=RAGArgs(min_similarity=0.7))

        call_kwargs = mock_context_builder.facts_client.search_similar.call_args.kwargs
        assert call_kwargs["min_similarity"] == 0.7

    @pytest.mark.asyncio
    async def test_ask_with_rag_custom_model_name(
        self, mock_client, mock_context_builder, mock_embedder
    ):
        """Test that RAGArgs.model_name overrides embedder model."""
        mock_context_builder.facts_client.search_similar.return_value = []

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("Test", rag=RAGArgs(model_name="custom-embedding-model"))

        call_kwargs = mock_context_builder.facts_client.search_similar.call_args.kwargs
        assert call_kwargs["model_name"] == "custom-embedding-model"

    @pytest.mark.asyncio
    async def test_ask_with_rag_uses_embedder_model_by_default(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG uses embedder's model when model_name is None."""
        mock_context_builder.facts_client.search_similar.return_value = []

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("Test", rag=RAGArgs())

        call_kwargs = mock_context_builder.facts_client.search_similar.call_args.kwargs
        assert call_kwargs["model_name"] == "test-model"

    @pytest.mark.asyncio
    async def test_ask_with_rag_and_conversation(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG works with multi-turn conversations."""
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=sample_fact, similarity=0.9)
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        conv = Conversation()
        await query.ask("First question", conversation=conv, rag=RAGArgs())
        await query.ask("Follow up", conversation=conv, rag=RAGArgs())

        # Conversation should have both exchanges
        assert len(conv.messages) == 4
        assert conv.messages[0]["content"] == "First question"
        assert conv.messages[1]["content"] == "Test response"
        assert conv.messages[2]["content"] == "Follow up"
        assert conv.messages[3]["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_ask_with_rag_no_facts_found(
        self, mock_client, mock_context_builder, mock_embedder
    ):
        """Test RAG behavior when no similar facts are found."""
        mock_context_builder.facts_client.search_similar.return_value = []

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        await query.ask("Test", rag=RAGArgs())

        mock_context_builder.build_system_prompt_from_facts.assert_called_once_with("", [])

    @pytest.mark.asyncio
    async def test_ask_without_rag_uses_static_retrieval(self, mock_client, mock_context_builder):
        """Test that ask() without RAG uses static fact retrieval."""
        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
        )

        await query.ask("Test question")

        mock_context_builder.build_system_prompt.assert_called_once()
        # Should not use RAG methods
        mock_context_builder.facts_client.search_similar.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_with_rag_ignores_include_facts(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG ignores include_facts parameter."""
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=sample_fact, similarity=0.9)
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        # Even with include_facts=False, RAG should work
        await query.ask("Test", include_facts=False, rag=RAGArgs())

        # RAG was used, not static retrieval
        mock_context_builder.build_system_prompt_from_facts.assert_called_once()
        mock_context_builder.build_system_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_with_rag_ignores_fact_categories(
        self, mock_client, mock_context_builder, mock_embedder, sample_fact
    ):
        """Test that RAG ignores fact_categories parameter."""
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=sample_fact, similarity=0.9)
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        # fact_categories should be ignored when RAG is used
        await query.ask("Test", fact_categories=["preferences"], rag=RAGArgs())

        # RAG was used, not static retrieval with categories
        mock_context_builder.build_system_prompt_from_facts.assert_called_once()
        mock_context_builder.build_system_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_with_rag_passes_categories_to_search(
        self, mock_client, mock_context_builder, mock_embedder
    ):
        """Test that RAGArgs.categories is passed to search_similar for SQL filtering."""
        pref_fact = Fact(
            id=1,
            profile_id=1,
            content="User prefers Python",
            category="preferences",
            source="user",
            confidence=1.0,
            active=True,
        )

        # Simulate SQL filtering - only preferences facts returned
        mock_context_builder.facts_client.search_similar.return_value = [
            ScoredFact(fact=pref_fact, similarity=0.9),
        ]

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        # Filter to only preferences category
        await query.ask("Test", rag=RAGArgs(categories=["preferences"]))

        # Categories should be passed to search_similar for SQL-level filtering
        call_kwargs = mock_context_builder.facts_client.search_similar.call_args.kwargs
        assert call_kwargs["categories"] == ["preferences"]

        # All facts from search should be passed to build_system_prompt_from_facts
        call_args = mock_context_builder.build_system_prompt_from_facts.call_args
        facts_passed = call_args[0][1]
        assert len(facts_passed) == 1
        assert facts_passed[0].category == "preferences"

    @pytest.mark.asyncio
    async def test_ask_with_rag_categories_empty_result(
        self, mock_client, mock_context_builder, mock_embedder
    ):
        """Test RAG when SQL category filter returns no results."""
        # Simulate SQL filtering returning no results
        mock_context_builder.facts_client.search_similar.return_value = []

        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            embedder=mock_embedder,
        )

        # Filter to a category with no matching facts
        await query.ask("Test", rag=RAGArgs(categories=["rules"]))

        # Categories should be passed to search_similar
        call_kwargs = mock_context_builder.facts_client.search_similar.call_args.kwargs
        assert call_kwargs["categories"] == ["rules"]

        # Empty list passed to build_system_prompt_from_facts
        call_args = mock_context_builder.build_system_prompt_from_facts.call_args
        facts_passed = call_args[0][1]
        assert len(facts_passed) == 0


class TestContextQueryBasic:
    """Test basic ContextQuery functionality (non-RAG)."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.chat_async = AsyncMock(return_value="Response")
        client.aclose = AsyncMock()
        return client

    @pytest.fixture
    def mock_context_builder(self):
        """Create mock context builder."""
        builder = MagicMock()
        builder.build_system_prompt.return_value = "System with facts"
        return builder

    @pytest.mark.asyncio
    async def test_ask_basic(self, mock_client, mock_context_builder):
        """Test basic ask without RAG."""
        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
        )

        response = await query.ask("Hello")

        assert response == "Response"
        mock_client.chat_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_without_facts(self, mock_client, mock_context_builder):
        """Test ask with include_facts=False."""
        query = ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
            base_system_prompt="Base prompt",
        )

        await query.ask("Hello", include_facts=False)

        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["system"] == "Base prompt"
        mock_context_builder.build_system_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_client, mock_context_builder):
        """Test async context manager."""
        async with ContextQuery(
            client=mock_client,
            context_builder=mock_context_builder,
        ) as query:
            await query.ask("Test")

        mock_client.aclose.assert_called_once()


class TestBuildSystemPromptFromFacts:
    """Unit tests for ContextBuilder.build_system_prompt_from_facts method."""

    @pytest.fixture
    def mock_facts_client(self):
        """Create a mock facts client."""
        return MagicMock()

    @pytest.fixture
    def sample_facts(self):
        """Create sample facts for testing."""
        return [
            Fact(
                id=1,
                profile_id=1,
                content="Fact one",
                category="preferences",
                source="user",
                confidence=1.0,
                active=True,
            ),
            Fact(
                id=2,
                profile_id=1,
                content="Fact two",
                category="preferences",
                source="user",
                confidence=1.0,
                active=True,
            ),
        ]

    def test_builds_prompt_with_facts(self, mock_facts_client, sample_facts):
        """Test building prompt from provided facts."""
        from llm_learn.inference.context import ContextBuilder

        builder = ContextBuilder(mock_facts_client)
        result = builder.build_system_prompt_from_facts(
            base_prompt="You are an assistant.",
            facts=sample_facts,
        )

        assert "You are an assistant." in result
        assert "About the user:" in result
        assert "Fact one" in result
        assert "Fact two" in result

    def test_builds_prompt_with_empty_facts(self, mock_facts_client):
        """Test building prompt with no facts returns base prompt."""
        from llm_learn.inference.context import ContextBuilder

        builder = ContextBuilder(mock_facts_client)
        result = builder.build_system_prompt_from_facts(
            base_prompt="Base prompt only",
            facts=[],
        )

        assert result == "Base prompt only"

    def test_builds_prompt_prepend(self, mock_facts_client):
        """Test prepending facts to prompt."""
        from llm_learn.inference.context import ContextBuilder

        fact = Fact(
            id=1,
            profile_id=1,
            content="Test fact",
            category="preferences",
            source="user",
            confidence=1.0,
            active=True,
        )

        builder = ContextBuilder(mock_facts_client)
        result = builder.build_system_prompt_from_facts(
            base_prompt="Base prompt",
            facts=[fact],
            fact_position="prepend",
        )

        # Facts should come before base prompt
        facts_pos = result.find("About the user:")
        base_pos = result.find("Base prompt")
        assert facts_pos < base_pos

    def test_groups_facts_by_category(self, mock_facts_client):
        """Test facts are grouped by category."""
        from llm_learn.inference.context import ContextBuilder

        facts = [
            Fact(
                id=1,
                profile_id=1,
                content="Pref 1",
                category="preferences",
                source="user",
                confidence=1.0,
                active=True,
            ),
            Fact(
                id=2,
                profile_id=1,
                content="Pref 2",
                category="preferences",
                source="user",
                confidence=1.0,
                active=True,
            ),
            Fact(
                id=3,
                profile_id=1,
                content="Rule 1",
                category="rules",
                source="user",
                confidence=1.0,
                active=True,
            ),
        ]

        builder = ContextBuilder(mock_facts_client)
        result = builder.build_system_prompt_from_facts("", facts)

        assert "### Preferences" in result
        assert "### Rules" in result
