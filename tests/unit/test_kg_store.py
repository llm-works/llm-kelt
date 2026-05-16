"""Unit tests for Knowledge Graph store."""

from __future__ import annotations

import pytest

from llm_kelt.memory.kg.store import scope_ancestors


@pytest.fixture(autouse=True)
def clean_kg_tables(database):
    """Clean KG tables before each test for isolation."""
    from llm_kelt.memory.kg.models import (
        Entity,
        EntityAlias,
        EntityRef,
        EntityRelationship,
        FactEntity,
    )

    with database.session() as session:
        # Delete in reverse FK order
        session.query(EntityRef).delete()
        session.query(FactEntity).delete()
        session.query(EntityRelationship).delete()
        session.query(EntityAlias).delete()
        session.query(Entity).delete()
        session.commit()
    yield


class TestScopeAncestors:
    """Tests for scope_ancestors helper function."""

    def test_global_scope(self):
        """Global scope returns only itself."""
        assert scope_ancestors("global") == ["global"]

    def test_single_level(self):
        """Single level scope includes global."""
        result = scope_ancestors("org:acme")
        assert "org:acme" in result
        assert "global" in result

    def test_two_levels(self):
        """Two level scope includes parent and global."""
        result = scope_ancestors("org:acme:user:alice")
        assert result == ["org:acme:user:alice", "org:acme", "global"]

    def test_three_levels(self):
        """Three level scope includes all ancestors."""
        result = scope_ancestors("org:acme:team:eng:user:bob")
        assert "org:acme:team:eng:user:bob" in result
        assert "org:acme:team:eng" in result
        assert "org:acme" in result
        assert "global" in result

    def test_order_is_specific_to_general(self):
        """Ancestors are ordered from most specific to most general."""
        result = scope_ancestors("org:acme:user:alice")
        assert result.index("org:acme:user:alice") < result.index("org:acme")
        assert result.index("org:acme") < result.index("global")


class TestEntityStore:
    """Tests for EntityStore CRUD operations."""

    def test_create_entity(self, kelt_client):
        """Create an entity in a scope."""
        entity = kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Tesla",
            entity_type="company",
            description="Electric vehicle manufacturer",
        )
        assert entity.id is not None
        assert entity.canonical_name == "tesla"  # lowercased
        assert entity.entity_type == "company"
        assert entity.scope_key == "global"

    def test_create_entity_with_aliases(self, kelt_client):
        """Create entity with initial aliases."""
        entity = kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Tesla",
            entity_type="company",
            aliases=["TSLA", "Tesla Inc"],
        )
        assert entity.id is not None
        assert len(entity.aliases) == 3  # canonical + 2 aliases

    def test_get_entity_by_id(self, kelt_client):
        """Get entity by ID."""
        created = kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="SpaceX",
            entity_type="company",
        )
        fetched = kelt_client.kg.entities.get(created.id)
        assert fetched is not None
        assert fetched.canonical_name == "spacex"

    def test_get_nonexistent_entity(self, kelt_client):
        """Get returns None for nonexistent ID."""
        assert kelt_client.kg.entities.get(999999) is None

    def test_resolve_by_canonical_name(self, kelt_client):
        """Resolve entity by canonical name."""
        kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Apple",
            entity_type="company",
        )
        resolved = kelt_client.kg.entities.resolve("global", "Apple", "company")
        assert resolved is not None
        assert resolved.canonical_name == "apple"

    def test_resolve_by_alias(self, kelt_client):
        """Resolve entity by alias."""
        entity = kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Apple",
            entity_type="company",
        )
        kelt_client.kg.entities.add_alias(entity.id, "AAPL", scope_key="global")

        resolved = kelt_client.kg.entities.resolve("global", "AAPL", "company")
        assert resolved is not None
        assert resolved.canonical_name == "apple"

    def test_resolve_case_insensitive(self, kelt_client):
        """Resolution is case-insensitive."""
        kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Microsoft",
            entity_type="company",
        )
        assert kelt_client.kg.entities.resolve("global", "MICROSOFT", "company") is not None
        assert kelt_client.kg.entities.resolve("global", "microsoft", "company") is not None

    def test_find_or_create_creates_new(self, kelt_client):
        """find_or_create creates entity if not found."""
        entity_id, created = kelt_client.kg.entities.find_or_create(
            scope_key="global",
            name="Netflix",
            entity_type="company",
        )
        assert created is True
        assert entity_id is not None

    def test_find_or_create_finds_existing(self, kelt_client):
        """find_or_create returns existing entity."""
        first_id, created1 = kelt_client.kg.entities.find_or_create(
            scope_key="global",
            name="Netflix",
            entity_type="company",
        )
        second_id, created2 = kelt_client.kg.entities.find_or_create(
            scope_key="global",
            name="Netflix",
            entity_type="company",
        )
        assert created1 is True
        assert created2 is False
        assert first_id == second_id

    def test_find_or_create_via_alias(self, kelt_client):
        """find_or_create finds via alias."""
        entity_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global",
            name="Amazon",
            entity_type="company",
        )
        kelt_client.kg.entities.add_alias(entity_id, "AMZN", scope_key="global")

        found_id, created = kelt_client.kg.entities.find_or_create(
            scope_key="global",
            name="AMZN",
            entity_type="company",
        )
        assert created is False
        assert found_id == entity_id

    def test_update_entity(self, kelt_client):
        """Update entity fields."""
        entity = kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Google",
            entity_type="company",
        )
        updated = kelt_client.kg.entities.update(
            entity.id,
            description="Search engine company",
            extra={"founded": 1998},
        )
        assert updated is not None
        assert updated.description == "Search engine company"
        assert updated.extra["founded"] == 1998

    def test_delete_entity(self, kelt_client):
        """Delete entity cascades."""
        entity = kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Defunct Corp",
            entity_type="company",
            aliases=["DC"],
        )
        entity_id = entity.id

        deleted = kelt_client.kg.entities.delete(entity_id)
        assert deleted is True
        assert kelt_client.kg.entities.get(entity_id) is None


class TestScopedQueries:
    """Tests for scope-based entity queries."""

    def test_in_scope_sees_own_scope(self, kelt_client):
        """in_scope returns entities in the specified scope."""
        kelt_client.kg.entities.create(
            scope_key="org:acme:user:alice",
            canonical_name="My Note",
            entity_type="note",
        )
        entities = kelt_client.kg.entities.in_scope("org:acme:user:alice")
        assert len(entities) == 1
        assert entities[0].canonical_name == "my note"

    def test_in_scope_sees_ancestors(self, kelt_client):
        """in_scope returns entities from ancestor scopes."""
        kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Tesla",
            entity_type="company",
        )
        kelt_client.kg.entities.create(
            scope_key="org:acme",
            canonical_name="Acme Project",
            entity_type="project",
        )

        # User scope should see both global and org entities
        entities = kelt_client.kg.entities.in_scope("org:acme:user:alice")
        names = {e.canonical_name for e in entities}
        assert "tesla" in names
        assert "acme project" in names

    def test_in_scope_does_not_see_siblings(self, kelt_client):
        """in_scope does not return entities from sibling scopes."""
        kelt_client.kg.entities.create(
            scope_key="org:acme:user:alice",
            canonical_name="Alice Secret",
            entity_type="note",
        )
        kelt_client.kg.entities.create(
            scope_key="org:acme:user:bob",
            canonical_name="Bob Secret",
            entity_type="note",
        )

        alice_entities = kelt_client.kg.entities.in_scope("org:acme:user:alice")
        names = {e.canonical_name for e in alice_entities}
        assert "alice secret" in names
        assert "bob secret" not in names

    def test_resolve_checks_scope_hierarchy(self, kelt_client):
        """Resolution checks scope hierarchy."""
        kelt_client.kg.entities.create(
            scope_key="global",
            canonical_name="Tesla",
            entity_type="company",
        )

        # Should resolve from user scope (via global ancestor)
        resolved = kelt_client.kg.entities.resolve("org:acme:user:alice", "Tesla", "company")
        assert resolved is not None


class TestEntityRelationships:
    """Tests for entity-to-entity relationships."""

    def test_add_relationship(self, kelt_client):
        """Add relationship between entities."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        elon_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Elon Musk", entity_type="person"
        )

        rel = kelt_client.kg.relationships.add(
            from_entity_id=tesla_id,
            to_entity_id=elon_id,
            relationship_type="founded_by",
            scope_key="global",
        )
        assert rel.id is not None
        assert rel.relationship_type == "founded_by"

    def test_get_relationships(self, kelt_client):
        """Get relationships for an entity."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        elon_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Elon Musk", entity_type="person"
        )
        spacex_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="SpaceX", entity_type="company"
        )

        kelt_client.kg.relationships.add(tesla_id, elon_id, "founded_by", scope_key="global")
        kelt_client.kg.relationships.add(spacex_id, elon_id, "founded_by", scope_key="global")

        rels = kelt_client.kg.relationships.get_relationships(
            elon_id, scope_key="global", direction="to"
        )
        assert len(rels) == 2

    def test_get_relationships_for_entities_batch(self, kelt_client):
        """Get relationships for multiple entities in one query."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        spacex_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="SpaceX", entity_type="company"
        )
        elon_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Elon Musk", entity_type="person"
        )

        kelt_client.kg.relationships.add(tesla_id, elon_id, "founded_by", scope_key="global")
        kelt_client.kg.relationships.add(spacex_id, elon_id, "founded_by", scope_key="global")

        results = kelt_client.kg.relationships.get_relationships_for_entities(
            [tesla_id, spacex_id], scope_key="global"
        )

        assert len(results) == 2
        assert len(results[tesla_id]) == 1
        assert len(results[spacex_id]) == 1
        assert results[tesla_id][0].to_entity_id == elon_id
        assert results[spacex_id][0].to_entity_id == elon_id

    def test_get_relationships_for_entities_empty(self, kelt_client):
        """Batch query with empty list returns empty dict."""
        results = kelt_client.kg.relationships.get_relationships_for_entities([], "global")
        assert results == {}


class TestEntityBatchQueries:
    """Tests for batch entity queries."""

    def test_get_by_names(self, kelt_client):
        """Get multiple entities by name in one query."""
        kelt_client.kg.entities.create(
            scope_key="global", canonical_name="Tesla", entity_type="company"
        )
        kelt_client.kg.entities.create(
            scope_key="global", canonical_name="SpaceX", entity_type="company"
        )
        kelt_client.kg.entities.create(
            scope_key="global", canonical_name="Apple", entity_type="company"
        )

        entities = kelt_client.kg.entities.get_by_names("global", ["Tesla", "SpaceX"], "company")

        assert len(entities) == 2
        names = {e.canonical_name for e in entities}
        assert "tesla" in names
        assert "spacex" in names
        assert "apple" not in names

    def test_get_by_names_case_insensitive(self, kelt_client):
        """Batch name lookup is case-insensitive."""
        kelt_client.kg.entities.create(
            scope_key="global", canonical_name="Tesla", entity_type="company"
        )

        entities = kelt_client.kg.entities.get_by_names(
            "global", ["TESLA", "tesla", "Tesla"], "company"
        )

        assert len(entities) == 1
        assert entities[0].canonical_name == "tesla"

    def test_get_by_names_empty_list(self, kelt_client):
        """Batch query with empty list returns empty list."""
        entities = kelt_client.kg.entities.get_by_names("global", [], "company")
        assert entities == []

    def test_get_by_names_not_found(self, kelt_client):
        """Batch query for nonexistent names returns empty."""
        entities = kelt_client.kg.entities.get_by_names(
            "global", ["NotExist1", "NotExist2"], "company"
        )
        assert entities == []


class TestFactEntityLinkage:
    """Tests for linking facts to entities."""

    def test_link_fact_to_entity(self, kelt_client):
        """Link a fact to an entity."""
        # Create a fact
        fact_id = kelt_client.atomic.assertions.add(
            content="Tesla is entering the robotics market",
            category="news",
        )
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )

        link = kelt_client.kg.fact_entities.link(
            fact_id=fact_id,
            entity_id=tesla_id,
            scope_key="global",
            role="subject",
        )
        assert link.fact_id == fact_id
        assert link.entity_id == tesla_id

    def test_get_entities_for_fact(self, kelt_client):
        """Get entities linked to a fact."""
        fact_id = kelt_client.atomic.assertions.add(
            content="Tesla acquired SolarCity",
            category="news",
        )
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        solarcity_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="SolarCity", entity_type="company"
        )

        kelt_client.kg.fact_entities.link(fact_id, tesla_id, "global", role="subject")
        kelt_client.kg.fact_entities.link(fact_id, solarcity_id, "global", role="object")

        entities = kelt_client.kg.fact_entities.get_entities_for_fact(fact_id, "global")
        assert len(entities) == 2
        names = {e[0].canonical_name for e in entities}
        assert "tesla" in names
        assert "solarcity" in names

    def test_get_facts_for_entity(self, kelt_client):
        """Get facts linked to an entity."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )

        fact1 = kelt_client.atomic.assertions.add("Tesla news 1", category="news")
        fact2 = kelt_client.atomic.assertions.add("Tesla news 2", category="news")

        kelt_client.kg.fact_entities.link(fact1, tesla_id, "global")
        kelt_client.kg.fact_entities.link(fact2, tesla_id, "global")

        fact_ids = kelt_client.kg.fact_entities.get_facts_for_entity(tesla_id, "global")
        assert len(fact_ids) == 2
        assert fact1 in fact_ids
        assert fact2 in fact_ids

    def test_get_entities_for_facts_batch(self, kelt_client):
        """Get entities for multiple facts in one query."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        spacex_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="SpaceX", entity_type="company"
        )
        source_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global",
            name="abc123hash",
            entity_type="source",
            extra={"url": "https://example.com/article"},
        )

        fact1 = kelt_client.atomic.assertions.add("Tesla news", category="news")
        fact2 = kelt_client.atomic.assertions.add("SpaceX news", category="news")
        fact3 = kelt_client.atomic.assertions.add("Unlinked fact", category="news")

        kelt_client.kg.fact_entities.link(fact1, tesla_id, "global", role="subject")
        kelt_client.kg.fact_entities.link(fact1, source_id, "global", role="cites")
        kelt_client.kg.fact_entities.link(fact2, spacex_id, "global", role="subject")
        kelt_client.kg.fact_entities.link(fact2, source_id, "global", role="cites")

        results = kelt_client.kg.fact_entities.get_entities_for_facts(
            [fact1, fact2, fact3], "global"
        )

        assert len(results) == 3
        assert len(results[fact1]) == 2
        assert len(results[fact2]) == 2
        assert len(results[fact3]) == 0

        fact1_names = {e.canonical_name for e, _, _ in results[fact1]}
        assert "tesla" in fact1_names
        assert "abc123hash" in fact1_names

    def test_get_entities_for_facts_filters_by_role(self, kelt_client):
        """Batch query filters by role."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        source_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="src123", entity_type="source"
        )

        fact1 = kelt_client.atomic.assertions.add("Tesla fact", category="news")
        kelt_client.kg.fact_entities.link(fact1, tesla_id, "global", role="subject")
        kelt_client.kg.fact_entities.link(fact1, source_id, "global", role="cites")

        results = kelt_client.kg.fact_entities.get_entities_for_facts(
            [fact1], "global", role="cites"
        )
        assert len(results[fact1]) == 1
        assert results[fact1][0][0].canonical_name == "src123"

    def test_get_entities_for_facts_filters_by_entity_type(self, kelt_client):
        """Batch query filters by entity type."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        source_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="src456", entity_type="source"
        )

        fact1 = kelt_client.atomic.assertions.add("Tesla fact", category="news")
        kelt_client.kg.fact_entities.link(fact1, tesla_id, "global", role="subject")
        kelt_client.kg.fact_entities.link(fact1, source_id, "global", role="cites")

        results = kelt_client.kg.fact_entities.get_entities_for_facts(
            [fact1], "global", entity_type="source"
        )
        assert len(results[fact1]) == 1
        assert results[fact1][0][0].entity_type == "source"

    def test_get_entities_for_facts_empty_list(self, kelt_client):
        """Batch query with empty list returns empty dict."""
        results = kelt_client.kg.fact_entities.get_entities_for_facts([], "global")
        assert results == {}


class TestEntityRefs:
    """Tests for entity reference tracking."""

    def test_add_ref(self, kelt_client):
        """Add reference to an entity."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )

        ref = kelt_client.kg.refs.add(
            entity_id=tesla_id,
            scope_key="org:acme:user:alice",
            source_type="article",
            source_id="article_123",
            snippet="Tesla announced new factory",
            sentiment=0.8,
        )
        assert ref.id is not None
        assert ref.sentiment == 0.8

    def test_count_refs(self, kelt_client):
        """Count references for an entity."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )

        kelt_client.kg.refs.add(tesla_id, "org:acme", "article", source_id="a1")
        kelt_client.kg.refs.add(tesla_id, "org:acme", "article", source_id="a2")
        kelt_client.kg.refs.add(tesla_id, "org:acme", "tweet", source_id="t1")

        count = kelt_client.kg.refs.count_by_entity(tesla_id)
        assert count == 3

    def test_trending_entities(self, kelt_client):
        """Get trending entities by reference count."""
        tesla_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Tesla", entity_type="company"
        )
        apple_id, _ = kelt_client.kg.entities.find_or_create(
            scope_key="global", name="Apple", entity_type="company"
        )

        # Tesla gets more refs
        for i in range(5):
            kelt_client.kg.refs.add(tesla_id, "org:acme", "article", source_id=f"t{i}")
        for i in range(2):
            kelt_client.kg.refs.add(apple_id, "org:acme", "article", source_id=f"a{i}")

        trending = kelt_client.kg.refs.trending("org:acme", limit=10)
        assert len(trending) == 2
        assert trending[0][0].canonical_name == "tesla"  # Most refs
        assert trending[0][1] == 5
        assert trending[1][0].canonical_name == "apple"
        assert trending[1][1] == 2


class TestEntityEmbeddings:
    """Tests for entity embedding operations."""

    @pytest.fixture(autouse=True)
    def clean_embeddings(self, database):
        """Clean embedding table before each test."""
        from llm_kelt.core.embedding import Embedding

        with database.session() as session:
            session.query(Embedding).delete()
            session.commit()
        yield

    @pytest.fixture
    def kg_with_embeddings(self, logger, database):
        """Create KGStore with embedding support (no embedder for manual tests)."""
        from llm_kelt.core.embedding import EmbeddingStore
        from llm_kelt.memory.kg import KGStore

        embedding_store = EmbeddingStore(database.session)
        return KGStore(
            logger,
            database.session,
            embedder=None,
            embedding_store=embedding_store,
        )

    def test_set_and_get_embedding(self, kg_with_embeddings):
        """Store and retrieve entity embedding."""
        entity = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="Tesla",
            entity_type="company",
        )

        embedding = [0.1, 0.2, 0.3]
        kg_with_embeddings.embeddings.set_embedding(
            entity_id=entity.id,
            embedding=embedding,
            model_name="test-model",
        )

        retrieved = kg_with_embeddings.embeddings.get_embedding(
            entity_id=entity.id,
            model_name="test-model",
        )
        assert retrieved is not None
        assert len(retrieved) == 3
        # Check approximate equality for floats
        for i, val in enumerate(embedding):
            assert abs(retrieved[i] - val) < 0.0001

    def test_get_nonexistent_embedding(self, kg_with_embeddings):
        """Get returns None for entity without embedding."""
        entity = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="NoEmbed",
            entity_type="company",
        )

        result = kg_with_embeddings.embeddings.get_embedding(
            entity_id=entity.id,
            model_name="test-model",
        )
        assert result is None

    def test_delete_embedding(self, kg_with_embeddings):
        """Delete entity embedding."""
        entity = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="ToDelete",
            entity_type="company",
        )

        kg_with_embeddings.embeddings.set_embedding(
            entity_id=entity.id,
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
        )

        deleted_count = kg_with_embeddings.embeddings.delete_embedding(entity.id)
        assert deleted_count == 1

        # Verify it's gone
        result = kg_with_embeddings.embeddings.get_embedding(
            entity_id=entity.id,
            model_name="test-model",
        )
        assert result is None

    def test_search_similar_entities(self, kg_with_embeddings):
        """Search for similar entities by embedding."""
        # Create entities with embeddings
        tesla = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="Tesla",
            entity_type="company",
            description="Electric vehicle company",
        )
        apple = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="Apple",
            entity_type="company",
            description="Technology company",
        )
        spacex = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="SpaceX",
            entity_type="company",
            description="Space exploration company",
        )

        # Set embeddings - Tesla and SpaceX are more similar (Elon companies)
        kg_with_embeddings.embeddings.set_embedding(
            entity_id=tesla.id,
            embedding=[0.9, 0.1, 0.8],  # Similar to SpaceX
            model_name="test-model",
        )
        kg_with_embeddings.embeddings.set_embedding(
            entity_id=apple.id,
            embedding=[0.1, 0.9, 0.1],  # Different
            model_name="test-model",
        )
        kg_with_embeddings.embeddings.set_embedding(
            entity_id=spacex.id,
            embedding=[0.85, 0.15, 0.75],  # Similar to Tesla
            model_name="test-model",
        )

        # Search for entities similar to Tesla's embedding
        results = kg_with_embeddings.embeddings.search_similar(
            query_embedding=[0.9, 0.1, 0.8],
            scope_key="global",
            model_name="test-model",
            limit=3,
        )

        assert len(results) == 3
        # Tesla should be first (exact match), SpaceX second (similar)
        names = [e.canonical_name for e, _ in results]
        assert names[0] == "tesla"
        assert names[1] == "spacex"
        assert names[2] == "apple"

    def test_search_respects_scope(self, kg_with_embeddings):
        """Search only returns entities visible in scope."""
        # Create entity in global scope
        global_entity = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="GlobalCorp",
            entity_type="company",
        )
        # Create entity in org scope
        org_entity = kg_with_embeddings.entities.create(
            scope_key="org:acme",
            canonical_name="AcmeCorp",
            entity_type="company",
        )
        # Create entity in different org
        other_entity = kg_with_embeddings.entities.create(
            scope_key="org:other",
            canonical_name="OtherCorp",
            entity_type="company",
        )

        # Add embeddings
        for entity in [global_entity, org_entity, other_entity]:
            kg_with_embeddings.embeddings.set_embedding(
                entity_id=entity.id,
                embedding=[0.5, 0.5, 0.5],
                model_name="test-model",
            )

        # Search from org:acme scope - should see global and org:acme, not org:other
        results = kg_with_embeddings.embeddings.search_similar(
            query_embedding=[0.5, 0.5, 0.5],
            scope_key="org:acme",
            model_name="test-model",
            limit=10,
        )

        names = {e.canonical_name for e, _ in results}
        assert "globalcorp" in names
        assert "acmecorp" in names
        assert "othercorp" not in names

    def test_embed_entity_requires_embedder(self, kg_with_embeddings):
        """embed_entity raises error without embedder configured."""
        entity = kg_with_embeddings.entities.create(
            scope_key="global",
            canonical_name="Test",
            entity_type="company",
        )

        with pytest.raises(RuntimeError, match="No embedder configured"):
            kg_with_embeddings.embeddings.embed_entity(entity, "test-model")

    def test_embed_text_requires_embedder(self, kg_with_embeddings):
        """embed_text raises error without embedder configured."""
        with pytest.raises(RuntimeError, match="No embedder configured"):
            kg_with_embeddings.embeddings.embed_text("some query")

    def test_embeddings_not_configured_raises(self, logger, database):
        """Accessing embeddings without embedding_store raises error."""
        from llm_kelt.memory.kg import KGStore

        # Create KGStore without embedding_store
        kg = KGStore(logger, database.session, embedder=None, embedding_store=None)
        with pytest.raises(RuntimeError, match="Embeddings not configured"):
            _ = kg.embeddings
