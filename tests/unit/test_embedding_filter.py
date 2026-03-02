"""Unit tests for EmbeddingFilter."""

from sqlalchemy import select

from llm_kelt.memory.atomic import EmbeddingFilter, Fact


class TestEmbeddingFilter:
    """Tests for EmbeddingFilter class."""

    def test_empty_filter(self):
        """Empty filter should be falsy and build to None."""
        f = EmbeddingFilter()
        assert not f
        assert f.build() is None

    def test_fact_type_filter(self):
        """fact_type() should filter by Fact.type."""
        f = EmbeddingFilter().fact_type("solution")
        assert f
        clause = f.build()
        assert clause is not None
        # Verify it compiles to valid SQL
        stmt = select(Fact).where(clause)
        assert "atomic_facts.type" in str(stmt)

    def test_categories_filter(self):
        """categories() should filter by Fact.category IN (...)."""
        f = EmbeddingFilter().categories("joke", "riddle")
        assert f
        clause = f.build()
        assert clause is not None
        stmt = select(Fact).where(clause)
        compiled = str(stmt)
        assert "atomic_facts.category" in compiled
        assert "IN" in compiled

    def test_where_clause(self):
        """where() should add raw SQLAlchemy clause."""
        f = EmbeddingFilter().where(Fact.confidence > 0.8)
        assert f
        clause = f.build()
        assert clause is not None
        stmt = select(Fact).where(clause)
        assert "atomic_facts.confidence" in str(stmt)

    def test_combined_filters(self):
        """Multiple filters should be ANDed together."""
        f = EmbeddingFilter().fact_type("solution").categories("joke").where(Fact.confidence > 0.5)
        clause = f.build()
        assert clause is not None
        stmt = select(Fact).where(clause)
        compiled = str(stmt)
        assert "atomic_facts.type" in compiled
        assert "atomic_facts.category" in compiled
        assert "atomic_facts.confidence" in compiled

    def test_multiple_where_clauses(self):
        """Multiple where() calls should be ANDed."""
        f = EmbeddingFilter().where(Fact.confidence > 0.5).where(Fact.active == True)  # noqa: E712
        clause = f.build()
        assert clause is not None
        stmt = select(Fact).where(clause)
        compiled = str(stmt)
        assert "atomic_facts.confidence" in compiled
        assert "atomic_facts.active" in compiled

    def test_method_chaining(self):
        """All methods should return self for chaining."""
        f = EmbeddingFilter()
        assert f.fact_type("x") is f
        assert f.categories("y") is f
        assert f.where(Fact.id > 0) is f

    def test_repr_empty(self):
        """repr of empty filter."""
        f = EmbeddingFilter()
        assert repr(f) == "EmbeddingFilter()"

    def test_repr_with_filters(self):
        """repr with filters set."""
        f = EmbeddingFilter().fact_type("solution").categories("joke").where(Fact.id > 0)
        r = repr(f)
        assert "fact_type='solution'" in r
        assert "categories=['joke']" in r
        assert "clauses=1" in r

    def test_subquery_filter(self):
        """where() should support subqueries."""
        # Simulate filtering by a subquery (e.g., exclude certain fact IDs)
        subquery = select(Fact.id).where(Fact.source == "haiku").scalar_subquery()
        f = EmbeddingFilter().where(Fact.id.not_in(subquery))
        clause = f.build()
        assert clause is not None
        stmt = select(Fact).where(clause)
        compiled = str(stmt)
        assert "NOT IN" in compiled
