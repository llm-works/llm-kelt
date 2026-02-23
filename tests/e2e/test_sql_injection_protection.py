"""Tests for SQL injection protection in context_key filtering.

Verifies that context keys with SQL wildcard characters (_, %) are properly
escaped and don't cause unintended pattern matching.

This addresses the security fix from commit 0a8b5da where underscore in context
keys (e.g., "acme_prod") was treated as SQL LIKE wildcard character, breaking
tenant isolation.
"""

import pytest

from llm_learn import ClientContext


class TestSQLInjectionProtection:
    """Test SQL wildcard escaping in context_key filtering."""

    @pytest.fixture
    def client_acme_prod(self, logger, database):
        """Create client for acme_prod context (contains underscore)."""
        from llm_learn.client import LearnClient

        context = ClientContext(context_key="acme_prod")
        return LearnClient(database=database, context=context, lg=logger)

    @pytest.fixture
    def client_acme_dev(self, logger, database):
        """Create client for acme_dev context (different but similar key)."""
        from llm_learn.client import LearnClient

        context = ClientContext(context_key="acme_dev")
        return LearnClient(database=database, context=context, lg=logger)

    @pytest.fixture
    def client_acme_x_prod(self, logger, database):
        """Create client for acmeXprod context (X matches _ wildcard if not escaped)."""
        from llm_learn.client import LearnClient

        context = ClientContext(context_key="acmeXprod")
        return LearnClient(database=database, context=context, lg=logger)

    def test_underscore_not_treated_as_wildcard(
        self, client_acme_prod, client_acme_dev, client_acme_x_prod, clean_tables
    ):
        """Test that underscore in context_key is NOT treated as SQL wildcard.

        Without proper escaping, "acme_prod" would match "acmeXprod" because
        underscore acts as SQL wildcard (matches any single character).

        This test verifies that:
        1. "acme_prod" context only sees "acme_prod" data
        2. "acme_dev" context only sees "acme_dev" data
        3. "acmeXprod" context only sees "acmeXprod" data
        4. No cross-contamination despite similar patterns
        """
        # Add data to each context
        fact_prod = client_acme_prod.atomic.assertions.add("Production setting")
        fact_dev = client_acme_dev.atomic.assertions.add("Development setting")
        fact_x = client_acme_x_prod.atomic.assertions.add("X setting")

        # Each client should ONLY see its own data
        prod_facts = client_acme_prod.atomic.assertions.list()
        dev_facts = client_acme_dev.atomic.assertions.list()
        x_facts = client_acme_x_prod.atomic.assertions.list()

        # Verify isolation
        assert len(prod_facts) == 1
        assert prod_facts[0].content == "Production setting"
        assert prod_facts[0].id == fact_prod

        assert len(dev_facts) == 1
        assert dev_facts[0].content == "Development setting"
        assert dev_facts[0].id == fact_dev

        assert len(x_facts) == 1
        assert x_facts[0].content == "X setting"
        assert x_facts[0].id == fact_x

    def test_percent_not_treated_as_wildcard(self, logger, database, clean_tables):
        """Test that percent sign in context_key is NOT treated as SQL wildcard."""
        from llm_learn.client import LearnClient

        # Create clients with percent sign in context key
        context_percent = ClientContext(context_key="discount_25%")
        context_other = ClientContext(context_key="discount_25X")

        client_percent = LearnClient(database=database, context=context_percent, lg=logger)
        client_other = LearnClient(database=database, context=context_other, lg=logger)

        # Add data
        fact_percent = client_percent.atomic.assertions.add("25% discount setting")
        fact_other = client_other.atomic.assertions.add("25X discount setting")

        # Each client should ONLY see its own data
        percent_facts = client_percent.atomic.assertions.list()
        other_facts = client_other.atomic.assertions.list()

        assert len(percent_facts) == 1
        assert percent_facts[0].id == fact_percent

        assert len(other_facts) == 1
        assert other_facts[0].id == fact_other

    def test_glob_patterns_work_correctly(self, logger, database, clean_tables):
        """Test that glob patterns (* and ?) work as intended after escaping.

        Verifies that:
        1. Literal underscores are escaped and match exactly
        2. Glob wildcards (* and ?) work as expected
        3. No cross-contamination between literal and pattern matching
        """
        from llm_learn.client import LearnClient

        # Create multiple contexts with hierarchy
        contexts = [
            "customer_123:prod:agent_a",
            "customer_123:prod:agent_b",
            "customer_123:dev:agent_a",
            "customer_456:prod:agent_a",
        ]

        clients = {}
        for ctx_key in contexts:
            context = ClientContext(context_key=ctx_key)
            clients[ctx_key] = LearnClient(database=database, context=context, lg=logger)

        # Add data to each context
        for ctx_key, client in clients.items():
            client.atomic.assertions.add(f"Data for {ctx_key}")

        # Test exact match - should see only one
        exact_client = LearnClient(
            database=database,
            context=ClientContext(context_key="customer_123:prod:agent_a"),
            lg=logger,
        )
        exact_facts = exact_client.atomic.assertions.list()
        assert len(exact_facts) == 1
        assert "customer_123:prod:agent_a" in exact_facts[0].content

        # Test glob pattern - all agents in customer_123:prod environment
        pattern_client = LearnClient(
            database=database,
            context=ClientContext(context_key="customer_123:prod:*"),
            lg=logger,
        )
        pattern_facts = pattern_client.atomic.assertions.list()
        assert len(pattern_facts) == 2  # agent_a and agent_b
        contents = [f.content for f in pattern_facts]
        assert any("agent_a" in c for c in contents)
        assert any("agent_b" in c for c in contents)

        # Test broader glob - all customer_123 contexts
        broader_client = LearnClient(
            database=database,
            context=ClientContext(context_key="customer_123:*"),
            lg=logger,
        )
        broader_facts = broader_client.atomic.assertions.list()
        assert len(broader_facts) == 3  # prod:agent_a, prod:agent_b, dev:agent_a

        # Verify customer_456 data is NOT included (isolation maintained)
        customer_456_included = any("customer_456" in f.content for f in broader_facts)
        assert not customer_456_included

    def test_edge_cases_combined_wildcards(self, logger, database, clean_tables):
        """Test edge cases with multiple wildcard characters."""
        from llm_learn.client import LearnClient

        # Test data with multiple underscores and percent signs
        test_keys = [
            "test__double",  # Double underscore
            "test%%double",  # Double percent
            "test_X_mixed",  # Mixed underscores
            "test_%_combo",  # Underscore and percent
        ]

        clients = {}
        for key in test_keys:
            context = ClientContext(context_key=key)
            clients[key] = LearnClient(database=database, context=context, lg=logger)
            clients[key].atomic.assertions.add(f"Data for {key}")

        # Verify each client sees only its own data (exact match)
        for key, client in clients.items():
            facts = client.atomic.assertions.list()
            assert len(facts) == 1
            assert key in facts[0].content
            # Verify no cross-contamination
            for other_key in test_keys:
                if other_key != key:
                    assert other_key not in facts[0].content
