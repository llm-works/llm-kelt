"""Tests for fact relationships (graph edges between atomic facts)."""

import pytest

from llm_kelt import Client, ClientContext, ConflictError, ValidationError
from llm_kelt.memory.atomic import RelType


class TestRelationshipsClient:
    """Test RelationshipsClient functionality."""

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _create_facts(kelt_client, n: int = 3) -> list[int]:
        """Create n assertion facts and return their IDs."""
        ids = []
        for i in range(n):
            fid = kelt_client.atomic.assertions.add(f"Test fact {i}")
            ids.append(fid)
        return ids

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------

    def test_link_creates_relationship(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        rel_id = kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)

        assert rel_id > 0

        rels = kelt_client.atomic.relationships.get_related(a, RelType.SUPPORTS)
        assert len(rels) == 1
        assert rels[0].source_id == a
        assert rels[0].target_id == b
        assert rels[0].relationship_type == "supports"
        assert rels[0].confidence == 1.0

    def test_link_with_metadata(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        meta = {"reason": "temporal overlap", "agent": "researcher"}
        kelt_client.atomic.relationships.link(a, b, RelType.CONTRADICTS, metadata=meta)

        rels = kelt_client.atomic.relationships.get_related(a)
        assert len(rels) == 1
        assert rels[0].metadata_ == meta

    def test_link_with_confidence(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS, confidence=0.7)

        rels = kelt_client.atomic.relationships.get_related(a)
        assert rels[0].confidence == pytest.approx(0.7)

    def test_link_default_confidence(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)

        rels = kelt_client.atomic.relationships.get_related(a)
        assert rels[0].confidence == 1.0

    def test_link_null_confidence(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS, confidence=None)

        rels = kelt_client.atomic.relationships.get_related(a)
        assert rels[0].confidence is None

    def test_unlink_specific_type(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        kelt_client.atomic.relationships.link(a, b, RelType.DERIVED_FROM)

        count = kelt_client.atomic.relationships.unlink(a, b, RelType.SUPPORTS)
        assert count == 1

        rels = kelt_client.atomic.relationships.get_related(a)
        assert len(rels) == 1
        assert rels[0].relationship_type == "derived_from"

    def test_unlink_all_types(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        kelt_client.atomic.relationships.link(a, b, RelType.DERIVED_FROM)

        count = kelt_client.atomic.relationships.unlink(a, b)
        assert count == 2

        rels = kelt_client.atomic.relationships.get_related(a)
        assert len(rels) == 0

    def test_unlink_nonexistent_returns_zero(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        count = kelt_client.atomic.relationships.unlink(a, b, RelType.SUPPORTS)
        assert count == 0

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def test_link_self_raises(self, kelt_client, clean_tables):
        a = self._create_facts(kelt_client, 1)[0]
        with pytest.raises(ValidationError, match="itself"):
            kelt_client.atomic.relationships.link(a, a, RelType.SUPPORTS)

    def test_link_invalid_confidence_raises(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        with pytest.raises(ValidationError, match="confidence must be"):
            kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS, confidence=1.5)

        with pytest.raises(ValidationError, match="confidence must be"):
            kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS, confidence=-0.1)

    def test_link_inactive_fact_raises(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.assertions.deactivate(b)

        with pytest.raises(ValidationError, match="not found or inactive"):
            kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)

    def test_link_nonexistent_fact_raises(self, kelt_client, clean_tables):
        a = self._create_facts(kelt_client, 1)[0]
        with pytest.raises(ValidationError, match="not found or inactive"):
            kelt_client.atomic.relationships.link(a, 999999, RelType.SUPPORTS)

    def test_link_duplicate_raises_conflict(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)

        with pytest.raises(ConflictError, match="already exists"):
            kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)

    # -------------------------------------------------------------------------
    # Symmetric behavior
    # -------------------------------------------------------------------------

    def test_symmetric_link_normalized(self, kelt_client, clean_tables):
        """Symmetric types store (min, max) — linking (B, A) after (A, B) is a conflict."""
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.CONTRADICTS)

        with pytest.raises(ConflictError):
            kelt_client.atomic.relationships.link(b, a, RelType.CONTRADICTS)

    def test_symmetric_get_related_either_direction(self, kelt_client, clean_tables):
        """Both facts can find the symmetric edge."""
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.CONTRADICTS)

        rels_a = kelt_client.atomic.relationships.get_related(a, RelType.CONTRADICTS)
        rels_b = kelt_client.atomic.relationships.get_related(b, RelType.CONTRADICTS)

        assert len(rels_a) == 1
        assert len(rels_b) == 1
        assert rels_a[0].id == rels_b[0].id

    def test_symmetric_unlink_either_direction(self, kelt_client, clean_tables):
        """Unlinking with either ID order works for symmetric types."""
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.RELATED_TO)

        # Unlink using reversed order
        count = kelt_client.atomic.relationships.unlink(b, a, RelType.RELATED_TO)
        assert count == 1

    # -------------------------------------------------------------------------
    # Directional behavior
    # -------------------------------------------------------------------------

    def test_directional_both_directions_allowed(self, kelt_client, clean_tables):
        """Directional types allow (A->B) and (B->A) as distinct edges."""
        a, b = self._create_facts(kelt_client, 2)
        id1 = kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        id2 = kelt_client.atomic.relationships.link(b, a, RelType.SUPPORTS)

        assert id1 != id2

    def test_get_related_outgoing(self, kelt_client, clean_tables):
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        kelt_client.atomic.relationships.link(c, a, RelType.SUPPORTS)

        rels = kelt_client.atomic.relationships.get_related(
            a, RelType.SUPPORTS, direction="outgoing"
        )
        assert len(rels) == 1
        assert rels[0].target_id == b

    def test_get_related_incoming(self, kelt_client, clean_tables):
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        kelt_client.atomic.relationships.link(c, a, RelType.SUPPORTS)

        rels = kelt_client.atomic.relationships.get_related(
            a, RelType.SUPPORTS, direction="incoming"
        )
        assert len(rels) == 1
        assert rels[0].source_id == c

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def test_find_contradictions_specific_fact(self, kelt_client, clean_tables):
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.CONTRADICTS)
        kelt_client.atomic.relationships.link(a, c, RelType.SUPPORTS)

        contras = kelt_client.atomic.relationships.find_contradictions(a)
        assert len(contras) == 1
        assert contras[0].relationship_type == "contradicts"

    def test_find_contradictions_all(self, kelt_client, clean_tables):
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.CONTRADICTS)
        kelt_client.atomic.relationships.link(a, c, RelType.SUPPORTS)

        contras = kelt_client.atomic.relationships.find_contradictions()
        assert len(contras) == 1

    def test_get_related_filters_by_type(self, kelt_client, clean_tables):
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        kelt_client.atomic.relationships.link(a, b, RelType.DERIVED_FROM)

        rels = kelt_client.atomic.relationships.get_related(a, RelType.SUPPORTS)
        assert len(rels) == 1
        assert rels[0].relationship_type == "supports"

    def test_get_related_loads_facts(self, kelt_client, clean_tables):
        """Verify that source_fact and target_fact are eagerly loaded."""
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)

        rels = kelt_client.atomic.relationships.get_related(a)
        assert rels[0].source_fact is not None
        assert rels[0].target_fact is not None
        assert rels[0].source_fact.id == a
        assert rels[0].target_fact.id == b

    def test_count(self, kelt_client, clean_tables):
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        kelt_client.atomic.relationships.link(a, c, RelType.CONTRADICTS)

        assert kelt_client.atomic.relationships.count() == 2
        assert kelt_client.atomic.relationships.count(RelType.SUPPORTS) == 1
        assert kelt_client.atomic.relationships.count(RelType.CONTRADICTS) == 1
        assert kelt_client.atomic.relationships.count(RelType.DERIVED_FROM) == 0

    # -------------------------------------------------------------------------
    # Chain / CTE
    # -------------------------------------------------------------------------

    def test_get_chain_simple(self, kelt_client, clean_tables):
        """A derived_from B derived_from C."""
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.DERIVED_FROM)
        kelt_client.atomic.relationships.link(b, c, RelType.DERIVED_FROM)

        chain = kelt_client.atomic.relationships.get_chain(a, RelType.DERIVED_FROM)
        assert len(chain) == 2
        # Ordered by depth: a->b first, then b->c
        assert chain[0].source_id == a
        assert chain[0].target_id == b
        assert chain[1].source_id == b
        assert chain[1].target_id == c

    def test_get_chain_max_depth(self, kelt_client, clean_tables):
        """Chain of 5 levels, limited to depth 2."""
        facts = self._create_facts(kelt_client, 6)
        for i in range(5):
            kelt_client.atomic.relationships.link(facts[i], facts[i + 1], RelType.DERIVED_FROM)

        chain = kelt_client.atomic.relationships.get_chain(
            facts[0], RelType.DERIVED_FROM, max_depth=2
        )
        assert len(chain) == 2

    def test_get_chain_max_results(self, kelt_client, clean_tables):
        """Wide graph limited by max_results."""
        facts = self._create_facts(kelt_client, 6)
        for i in range(5):
            kelt_client.atomic.relationships.link(facts[i], facts[i + 1], RelType.DERIVED_FROM)

        chain = kelt_client.atomic.relationships.get_chain(
            facts[0], RelType.DERIVED_FROM, max_results=2
        )
        assert len(chain) == 2

    def test_get_chain_cycle_detection(self, kelt_client, clean_tables):
        """A->B->C->A cycle terminates without infinite loop."""
        a, b, c = self._create_facts(kelt_client, 3)
        kelt_client.atomic.relationships.link(a, b, RelType.DERIVED_FROM)
        kelt_client.atomic.relationships.link(b, c, RelType.DERIVED_FROM)
        kelt_client.atomic.relationships.link(c, a, RelType.DERIVED_FROM)

        chain = kelt_client.atomic.relationships.get_chain(a, RelType.DERIVED_FROM)
        # Should return 3 edges without looping forever
        assert len(chain) == 3

    def test_get_chain_empty(self, kelt_client, clean_tables):
        a = self._create_facts(kelt_client, 1)[0]
        chain = kelt_client.atomic.relationships.get_chain(a, RelType.DERIVED_FROM)
        assert chain == []

    def test_get_chain_invalid_depth_raises(self, kelt_client, clean_tables):
        a = self._create_facts(kelt_client, 1)[0]
        with pytest.raises(ValidationError, match="max_depth"):
            kelt_client.atomic.relationships.get_chain(a, max_depth=0)

    # -------------------------------------------------------------------------
    # Context isolation
    # -------------------------------------------------------------------------

    def test_context_isolation(self, logger, database, clean_tables):
        """Different contexts don't see each other's relationships."""
        ctx_a = ClientContext(context_key="test_ctx_a", schema_name=None)
        ctx_b = ClientContext(context_key="test_ctx_b", schema_name=None)
        client_a = Client(database=database, context=ctx_a, lg=logger)
        client_b = Client(database=database, context=ctx_b, lg=logger)

        # Create facts in each context
        a1 = client_a.atomic.assertions.add("Fact A1")
        a2 = client_a.atomic.assertions.add("Fact A2")
        b1 = client_b.atomic.assertions.add("Fact B1")
        b2 = client_b.atomic.assertions.add("Fact B2")

        # Link in context A
        client_a.atomic.relationships.link(a1, a2, RelType.SUPPORTS)
        # Link in context B
        client_b.atomic.relationships.link(b1, b2, RelType.SUPPORTS)

        # Each context only sees its own
        assert client_a.atomic.relationships.count() == 1
        assert client_b.atomic.relationships.count() == 1
        assert len(client_a.atomic.relationships.get_related(a1)) == 1
        assert len(client_b.atomic.relationships.get_related(b1)) == 1

    # -------------------------------------------------------------------------
    # Cascade delete
    # -------------------------------------------------------------------------

    def test_cascade_on_fact_delete(self, kelt_client, clean_tables):
        """Deleting a fact cascades to its relationship rows."""
        a, b = self._create_facts(kelt_client, 2)
        kelt_client.atomic.relationships.link(a, b, RelType.SUPPORTS)
        assert kelt_client.atomic.relationships.count() == 1

        kelt_client.atomic.assertions.delete(a)
        assert kelt_client.atomic.relationships.count() == 0


class TestRelType:
    """Test RelType enum."""

    def test_from_value(self):
        assert RelType.from_value("contradicts") == RelType.CONTRADICTS
        assert RelType.from_value("supports") == RelType.SUPPORTS
        assert RelType.from_value("supersedes") == RelType.SUPERSEDES
        assert RelType.from_value("derived_from") == RelType.DERIVED_FROM
        assert RelType.from_value("related_to") == RelType.RELATED_TO

    def test_from_value_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown relationship type"):
            RelType.from_value("invalid")

    def test_symmetric_flag(self):
        assert RelType.CONTRADICTS.symmetric is True
        assert RelType.RELATED_TO.symmetric is True
        assert RelType.SUPPORTS.symmetric is False
        assert RelType.SUPERSEDES.symmetric is False
        assert RelType.DERIVED_FROM.symmetric is False

    def test_db_value(self):
        assert RelType.CONTRADICTS.db_value == "contradicts"
        assert RelType.SUPPORTS.db_value == "supports"
