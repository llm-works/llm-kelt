# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for Client.view() and ClientView."""

import threading
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_kelt import ClientContext, ClientView, SchemaMode, SchemaVersionError
from llm_kelt.client import Client


def _scoped_db(schema_name: str) -> Mock:
    scoped = Mock()
    scoped.schema = schema_name
    scoped.session = MagicMock()
    scoped.engine = Mock()
    return scoped


@pytest.fixture
def db() -> Mock:
    database = Mock()
    database.session = MagicMock()
    database.engine = Mock()
    database.schema = None
    database.scoped.side_effect = _scoped_db
    return database


def _client(lg, db, mode: SchemaMode = SchemaMode.SKIP, llm_client=None) -> Client:
    with patch.object(Client, "_apply_schema_mode"):
        return Client(
            lg,
            database=db,
            context=ClientContext(context_key="parent"),
            schema_mode=mode,
            embeddings=Mock(),
            llm_client=llm_client,
        )


class TestClientViewCreation:
    """Creating a view only binds the context."""

    def test_view_scopes_context_key(self, lg, db):
        client = _client(lg, db)

        view = client.view("agent-a")

        assert isinstance(view, ClientView)
        assert view.context_key == "agent-a"
        assert view.context.schema_name == client.context.schema_name
        assert client.context_key == "parent"

    def test_context_is_a_copy(self, lg, db):
        view = _client(lg, db).view("agent-a")

        with patch("llm_kelt.client_view.Protocol"):
            view.atomic
        view.context.context_key = "other"

        assert view.context_key == "agent-a"
        with patch("llm_kelt.client_view.ContentStore") as content:
            view.content
        assert content.call_args.args[1] == "agent-a"

    def test_view_creation_builds_nothing(self, lg, db):
        client = _client(lg, db)
        db.reset_mock()

        with (
            patch("llm_kelt.client.prepare_scoped_schema") as prepare,
            patch("llm_kelt.client_view.Protocol") as protocol,
        ):
            view = client.view("agent-a", schema_name="other")

        assert view.context.schema_name == "other"
        db.scoped.assert_not_called()
        prepare.assert_not_called()
        protocol.assert_not_called()


class TestClientViewStores:
    """Stores are built on first access with the view's context_key."""

    def test_atomic_uses_view_key_and_parent_session(self, lg, db):
        client = _client(lg, db)
        view = client.view("agent-a")

        with patch("llm_kelt.client_view.Protocol") as protocol:
            first = view.atomic
            second = view.atomic

        assert first is second
        protocol.assert_called_once()
        assert protocol.call_args.args[1] is db.session
        assert protocol.call_args.args[2] == "agent-a"

    def test_content_uses_view_key(self, lg, db):
        client = _client(lg, db)

        with patch("llm_kelt.client_view.ContentStore") as content:
            client.view("agent-a").content

        content.assert_called_once_with(db.session, "agent-a")

    def test_kg_on_parent_schema_is_parent_store(self, lg, db):
        client = _client(lg, db)

        assert client.view("agent-a").kg is client.kg

    def test_public_is_parent_schema_when_unset(self, lg, db):
        client = _client(lg, db)

        with patch("llm_kelt.client.prepare_scoped_schema") as prepare:
            kg = client.view("agent-a", schema_name="public").kg

        assert kg is client.kg
        db.scoped.assert_not_called()
        prepare.assert_not_called()

    def test_kg_on_other_schema_uses_scoped_session(self, lg, db):
        scoped = _scoped_db("other")
        db.scoped.side_effect = None
        db.scoped.return_value = scoped
        client = _client(lg, db)
        view = client.view("agent-a", schema_name="other")

        with (
            patch("llm_kelt.client.prepare_scoped_schema"),
            patch("llm_kelt.client.KGStore") as kg,
        ):
            view.kg

        assert kg.call_args.args[1] is scoped.session
        assert kg.call_args.kwargs["embedding_schema"] == "other"

    def test_views_on_parent_schema_share_embedding_stores(self, lg, db):
        client = _client(lg, db)

        with patch("llm_kelt.client_view.Protocol") as protocol:
            client.view("agent-a").atomic
            client.view("agent-b").atomic

        first, second = (call.kwargs["embedding_stores"] for call in protocol.call_args_list)
        assert first is second
        assert first is client.atomic.embeddings._stores

    def test_views_on_one_schema_share_its_scope(self, lg, db):
        client = _client(lg, db, mode=SchemaMode.VERIFY)

        with (
            patch("llm_kelt.client.prepare_scoped_schema") as prepare,
            patch("llm_kelt.client_view.Protocol") as protocol,
        ):
            first = client.view("agent-a", schema_name="other")
            second = client.view("agent-b", schema_name="other")
            first.atomic
            second.atomic
            kgs = (first.kg, second.kg)

        db.scoped.assert_called_once_with("other")
        prepare.assert_called_once()
        assert prepare.call_args.args[2:] == ("other", SchemaMode.VERIFY)
        stores = [call.kwargs["embedding_stores"] for call in protocol.call_args_list]
        assert stores[0] is stores[1]
        assert stores[0].schema == "other"
        assert kgs[0] is kgs[1]
        assert kgs[0] is not client.kg

    def test_stale_schema_raises_on_first_use(self, lg, db):
        client = _client(lg, db, mode=SchemaMode.VERIFY)
        view = client.view("agent-a", schema_name="other")

        with (
            patch(
                "llm_kelt.client.prepare_scoped_schema",
                side_effect=SchemaVersionError("not current"),
            ),
            pytest.raises(SchemaVersionError),
        ):
            view.atomic

    def test_query_requires_llm_client(self, lg, db):
        view = _client(lg, db).view("agent-a")

        with pytest.raises(RuntimeError, match="LLM client not configured"):
            view.query

    def test_query_uses_view_facts(self, lg, db):
        llm_client = Mock()
        view = _client(lg, db, llm_client=llm_client).view("agent-a")

        with patch("llm_kelt.client_view.ContextQuery") as query:
            view.query

        kwargs = query.call_args.kwargs
        assert kwargs["client"] is llm_client
        assert kwargs["context_builder"] is view.context_builder
        assert kwargs["embedding_adapter"] is view.atomic.embeddings


class TestClientViewConcurrency:
    """Views are shared across threads."""

    @pytest.mark.integration
    def test_threads_share_stores_and_one_schema_preparation(self, lg, db):
        client = _client(lg, db, mode=SchemaMode.VERIFY)
        shared = client.view("agent-shared", schema_name="other")
        workers = 16
        barrier = threading.Barrier(workers)
        seen: list[object] = []
        seen_lock = threading.Lock()

        def work(i: int) -> None:
            barrier.wait(timeout=10)
            client.view(f"agent-{i}", schema_name="other").atomic
            store = shared.atomic
            with seen_lock:
                seen.append(store)

        with patch("llm_kelt.client.prepare_scoped_schema") as prepare:
            threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert len(seen) == workers
        assert all(store is seen[0] for store in seen)
        db.scoped.assert_called_once_with("other")
        prepare.assert_called_once()
