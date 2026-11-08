# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Key patterns on facts and the knowledge graph: reads select, writes refuse."""

import pytest

from llm_kelt import Client, ClientContext, ValidationError
from llm_kelt.memory.atomic import RelType


@pytest.fixture
def root(test_context: str) -> str:
    """Key unique to the test; everything the test writes lives at or below it."""
    return f"t{test_context}"


@pytest.fixture
def client_at(logger, database):
    """Factory for clients opened at a key or pattern."""

    def make(key: str) -> Client:
        return Client(lg=logger, database=database, context=ClientContext(context_key=key))

    return make


class TestFactPatterns:
    """Facts and fact relationships under a key pattern."""

    def test_subtree_reads_key_and_below(self, client_at, root):
        """K:** sees K and keys below K:, not a sibling that shares K's prefix."""
        for key in ("org:a", "org:a:team:x", "org:ab", "org:b"):
            client_at(f"{root}:{key}").atomic.assertions.add(f"fact at {key}")

        facts = client_at(f"{root}:org:a:**").atomic.assertions.list_active()

        assert {f.content for f in facts} == {"fact at org:a", "fact at org:a:team:x"}

    def test_pattern_client_cannot_write(self, client_at, root):
        writer = client_at(f"{root}:org:a")
        first = writer.atomic.assertions.add("first")
        second = writer.atomic.assertions.add("second")
        reader = client_at(f"{root}:org:*")

        with pytest.raises(ValidationError, match="cannot be written"):
            reader.atomic.assertions.add("written through a pattern")
        with pytest.raises(ValidationError, match="cannot be written"):
            reader.atomic.relationships.link(first, second, RelType.SUPPORTS)

    def test_contradictions_scoped_by_pattern(self, client_at, root):
        for key in ("org:a:team:x", "org:b"):
            client = client_at(f"{root}:{key}")
            one = client.atomic.assertions.add(f"one at {key}")
            two = client.atomic.assertions.add(f"two at {key}")
            client.atomic.relationships.link(one, two, RelType.CONTRADICTS)

        found = client_at(f"{root}:org:a:**").atomic.relationships.find_contradictions()

        assert [r.context_key for r in found] == [f"{root}:org:a:team:x"]

    def test_chain_under_subtree(self, client_at, root):
        """get_chain's raw SQL applies the same pattern terms."""
        writer = client_at(f"{root}:org:a:team:x")
        a, b, c = (writer.atomic.assertions.add(t) for t in ("a", "b", "c"))
        writer.atomic.relationships.link(a, b, RelType.DERIVED_FROM)
        writer.atomic.relationships.link(b, c, RelType.DERIVED_FROM)

        chain = client_at(f"{root}:org:a:**").atomic.relationships.get_chain(a)

        assert [(r.source_id, r.target_id) for r in chain] == [(a, b), (b, c)]


class TestKGPatterns:
    """KG reads under a pattern: knowledge selects the subgraph and the entities it touches."""

    @pytest.fixture
    def graph(self, kelt_client, root):
        """Shared entities at the root; knowledge and local entities below it."""
        kg = kelt_client.kg
        team, other = f"{root}:org:a:team:x", f"{root}:org:b"
        company_a = kg.entities.create(root, "Company A", "company")
        company_b = kg.entities.create(root, "Company B", "company")
        unused = kg.entities.create(root, "Company C", "company")
        note = kg.entities.create(team, "Team Note", "note")
        other_note = kg.entities.create(other, "Other Note", "note")
        kg.refs.add(company_a.id, team, "doc")
        kg.relationships.add(company_a.id, company_b.id, "partners_with", team)
        kg.relationships.add(company_a.id, other_note.id, "mentioned_in", other)
        return {
            "company_a": company_a.id,
            "company_b": company_b.id,
            "unused": unused.id,
            "note": note.id,
            "other_note": other_note.id,
        }

    def test_in_scope_pattern(self, kelt_client, root, graph):
        """Entities created under matching keys, plus those matching knowledge touches."""
        found = {e.id for e in kelt_client.kg.entities.in_scope(f"{root}:org:a:**")}

        assert found == {graph["company_a"], graph["company_b"], graph["note"]}

    def test_in_scope_key_keeps_ancestors(self, kelt_client, root, graph):
        """A key still sees its own scope and every prefix of it."""
        found = {e.id for e in kelt_client.kg.entities.in_scope(f"{root}:org:a:team:x")}

        assert found == {graph["company_a"], graph["company_b"], graph["unused"], graph["note"]}

    def test_relationships_pattern(self, kelt_client, root, graph):
        rels = kelt_client.kg.relationships.get_relationships(
            graph["company_a"], f"{root}:org:a:**"
        )
        batch = kelt_client.kg.relationships.get_relationships_for_entities(
            [graph["company_a"]], f"{root}:org:a:**"
        )

        assert [r.to_entity_id for r in rels] == [graph["company_b"]]
        assert [r.to_entity_id for r in batch[graph["company_a"]]] == [graph["company_b"]]

    def test_entities_for_facts_pattern(self, kelt_client, client_at, root, graph):
        """The link's key selects; the entity may sit at any scope."""
        fact_a = client_at(f"{root}:org:a:team:x").atomic.assertions.add("fact a")
        fact_b = client_at(f"{root}:org:b").atomic.assertions.add("fact b")
        links = kelt_client.kg.fact_entities
        links.link(fact_a, graph["company_a"], f"{root}:org:a:team:x")
        links.link(fact_b, graph["company_b"], f"{root}:org:b")

        found = links.get_entities_for_facts([fact_a, fact_b], f"{root}:org:a:**")
        single = links.get_entities_for_fact(fact_a, f"{root}:org:a:**")

        assert [e.id for e, _, _ in found[fact_a]] == [graph["company_a"]]
        assert found[fact_b] == []
        assert [e.id for e, _, _ in single] == [graph["company_a"]]

    def test_other_reads_refuse_patterns(self, kelt_client, root, graph):
        kg = kelt_client.kg
        with pytest.raises(ValidationError, match="not a pattern"):
            kg.refs.trending(f"{root}:**")
        with pytest.raises(ValidationError, match="not a pattern"):
            kg.entities.resolve(f"{root}:**", "Company A")
        with pytest.raises(ValidationError, match="not a pattern"):
            kg.entities.find_or_create(f"{root}:**", "Company D", "company")

    def test_writes_refuse_patterns(self, kelt_client, client_at, root, graph):
        kg = kelt_client.kg
        fact = client_at(root).atomic.assertions.add("fact")
        with pytest.raises(ValidationError, match="cannot be written"):
            kg.entities.create(f"{root}:*", "Company D", "company")
        with pytest.raises(ValidationError, match="cannot be written"):
            kg.refs.add(graph["company_a"], f"{root}:**", "doc")
        with pytest.raises(ValidationError, match="cannot be written"):
            kg.fact_entities.link(fact, graph["company_a"], f"{root}:*", if_not_exists=True)
        with pytest.raises(ValidationError, match="cannot be written"):
            kg.entities.delete_in_scope(f"{root}:**")
