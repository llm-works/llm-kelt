"""Solutions client for agent problem/answer records."""

from typing import cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from llm_learn.core.exceptions import ValidationError

from ..models import Fact, SolutionDetails
from .base import FactClient


class SolutionsClient(FactClient[SolutionDetails]):
    """
    Client for recording agent solutions (problem/answer pairs).

    A solution represents an agent completing a task:
    - problem: What needed to be solved (the input)
    - answer: What the agent produced (the output)

    Usage:
        solutions = SolutionsClient(session_factory, context_key="my-agent")

        # Record a solution
        fact_id = solutions.record(
            agent_name="code-reviewer",
            problem="Review PR #123 for security issues",
            problem_context={"messages": [...], "tools": [...]},
            answer={"verdict": "approved", "comments": [...]},
            answer_text="PR approved with minor suggestions",
            tokens_used=1500,
            latency_ms=2340,
        )
    """

    fact_type = "solution"
    details_model = SolutionDetails
    details_relationship = "solution_details"

    def _validate_solution_inputs(
        self,
        agent_name: str,
        problem: str,
        problem_context: dict,
        answer: dict,
        tokens_used: int,
        latency_ms: int,
        tool_calls: list[dict] | None,
    ) -> None:
        """Validate solution record inputs."""
        if not agent_name or not agent_name.strip():
            raise ValidationError("agent_name cannot be empty")
        if not problem or not problem.strip():
            raise ValidationError("problem cannot be empty")
        if tokens_used < 0:
            raise ValidationError(f"tokens_used must be non-negative, got {tokens_used}")
        if latency_ms < 0:
            raise ValidationError(f"latency_ms must be non-negative, got {latency_ms}")
        if not isinstance(problem_context, dict):
            raise ValidationError("problem_context must be a dict")
        if not isinstance(answer, dict):
            raise ValidationError("answer must be a dict")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise ValidationError("tool_calls must be a list or None")
            if any(not isinstance(tc, dict) for tc in tool_calls):
                raise ValidationError("tool_calls elements must be dicts")

    def _create_fact(
        self,
        agent_name: str,
        problem: str,
        answer_text: str | None,
        category: str | None,
        source: str,
    ) -> Fact:
        """Create Fact record for solution.

        Uses answer_text if provided (for semantic search over solutions),
        otherwise falls back to problem summary.
        """
        # Use actual solution text if provided, otherwise summarize problem
        content_text = (
            answer_text
            if answer_text
            else f"{agent_name} solved: {problem[:100]}{'...' if len(problem) > 100 else ''}"
        )
        return Fact(
            context_key=self.context_key,
            type=self.fact_type,
            content=content_text,
            content_hash=self._compute_content_hash(content_text),
            category=category,
            source=source,
            confidence=1.0,
            active=True,
        )

    def _create_solution_details(
        self,
        fact_id: int,
        agent_name: str,
        problem: str,
        problem_context: dict,
        answer: dict,
        tokens_used: int,
        latency_ms: int,
        answer_text: str | None,
        tool_calls: list[dict] | None,
    ) -> SolutionDetails:
        """Create SolutionDetails record."""
        return SolutionDetails(
            fact_id=fact_id,
            agent_name=agent_name.strip(),
            problem=problem.strip(),
            problem_context=problem_context,
            answer=answer,
            answer_text=answer_text.strip() if answer_text else None,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
        )

    def record(
        self,
        agent_name: str,
        problem: str,
        problem_context: dict,
        answer: dict,
        tokens_used: int,
        latency_ms: int,
        answer_text: str | None = None,
        tool_calls: list[dict] | None = None,
        category: str | None = None,
        source: str = "agent",
    ) -> int:
        """Record an agent solution."""
        self._validate_solution_inputs(
            agent_name, problem, problem_context, answer, tokens_used, latency_ms, tool_calls
        )

        with self._session_factory() as session:
            fact = self._create_fact(agent_name, problem, answer_text, category, source)
            session.add(fact)
            session.flush()

            details = self._create_solution_details(
                fact.id,
                agent_name,
                problem,
                problem_context,
                answer,
                tokens_used,
                latency_ms,
                answer_text,
                tool_calls,
            )
            session.add(details)
            session.flush()

            # Auto-embed if embedder configured
            self._auto_embed_fact(fact, session)

            return fact.id

    def list_by_agent(
        self,
        agent_name: str,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Fact]:
        """List solutions for a specific agent."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(SolutionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    SolutionDetails.agent_name == agent_name,
                )
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.solution_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def list_by_category(
        self,
        category: str,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Fact]:
        """List solutions in a specific category."""
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.context_key == self.context_key,
                Fact.type == self.fact_type,
                Fact.category == category,
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.solution_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def get_agent_names(self) -> list[str]:
        """Get list of unique agent names with solutions."""
        with self._session_factory() as session:
            stmt = (
                select(SolutionDetails.agent_name)
                .join(Fact)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                )
                .distinct()
                .order_by(SolutionDetails.agent_name)
            )
            return list(session.scalars(stmt).all())

    def get_stats(self, agent_name: str | None = None) -> dict[str, int | float]:
        """Get statistics for solutions."""
        with self._session_factory() as session:
            from sqlalchemy import func as sqlfunc

            stmt = (
                select(
                    sqlfunc.count().label("count"),
                    sqlfunc.sum(SolutionDetails.tokens_used).label("total_tokens"),
                    sqlfunc.avg(SolutionDetails.latency_ms).label("avg_latency"),
                )
                .select_from(SolutionDetails)
                .join(Fact)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                )
            )

            if agent_name:
                stmt = stmt.where(SolutionDetails.agent_name == agent_name)

            result = session.execute(stmt).one()
            return {
                "count": result.count or 0,
                "total_tokens": result.total_tokens or 0,
                "avg_latency_ms": float(result.avg_latency) if result.avg_latency else 0.0,
            }

    def search(
        self,
        query: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> list[Fact]:
        """Search solutions by problem text."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(SolutionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    SolutionDetails.problem.ilike(f"%{query}%"),
                )
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.solution_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))
