"""Tests for auto-creation of profile chain on LearnClient init."""

from llm_learn.client import LearnClient
from llm_learn.core.profile import Profile
from llm_learn.core.workspace import Workspace


class TestEnsureProfile:
    """Test that LearnClient auto-creates profile and workspace when missing."""

    def test_auto_creates_profile_on_init(self, logger, database):
        """LearnClient should create the profile if it doesn't exist."""
        profile_id = Profile.generate_id(None, "default", "auto-test-1")

        client = LearnClient(logger, profile_id=profile_id, database=database)

        with database.session() as session:
            profile = session.get(Profile, profile_id)
            assert profile is not None
            assert profile.id == profile_id
            assert profile.active is True

        # Should be able to add facts without FK violation
        fact_id = client.assertions.add("test fact", category="test")
        assert fact_id > 0

    def test_auto_creates_default_workspace(self, logger, database):
        """Auto-created profile should be under a 'default' workspace."""
        profile_id = Profile.generate_id(None, "default", "auto-test-2")

        LearnClient(logger, profile_id=profile_id, database=database)

        workspace_id = Workspace.generate_id(None, "default")
        with database.session() as session:
            workspace = session.get(Workspace, workspace_id)
            assert workspace is not None
            assert workspace.slug == "default"
            assert workspace.name == "Default"

            profile = session.get(Profile, profile_id)
            assert profile.workspace_id == workspace_id

    def test_idempotent_existing_profile(self, logger, database, test_profile):
        """Should not recreate a profile that already exists."""
        with database.session() as session:
            original = session.get(Profile, test_profile)
            original_workspace_id = original.workspace_id
            original_name = original.name

        # Create client with existing profile — should not modify it
        LearnClient(logger, profile_id=test_profile, database=database)

        with database.session() as session:
            profile = session.get(Profile, test_profile)
            assert profile.workspace_id == original_workspace_id
            assert profile.name == original_name

    def test_multiple_profiles_share_default_workspace(self, logger, database):
        """Multiple auto-created profiles should share the same default workspace."""
        pid_a = Profile.generate_id(None, "default", "auto-test-multi-a")
        pid_b = Profile.generate_id(None, "default", "auto-test-multi-b")

        LearnClient(logger, profile_id=pid_a, database=database)
        LearnClient(logger, profile_id=pid_b, database=database)

        workspace_id = Workspace.generate_id(None, "default")
        with database.session() as session:
            pa = session.get(Profile, pid_a)
            pb = session.get(Profile, pid_b)
            assert pa.workspace_id == workspace_id
            assert pb.workspace_id == workspace_id
