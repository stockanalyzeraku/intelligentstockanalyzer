"""Short-term conversational state - within-session continuity only.

This is intentionally NOT persisted anywhere. It exists so a follow-up
question like "what about net profit?" can be understood as "same
company, same period as before" without the user repeating themselves.

Ownership model: the CALLER (CLI loop, web session handler, etc.) owns a
ConversationState instance and passes it into answer_query(), which
returns an updated one back. pipeline.py holds no hidden global state -
this keeps the pipeline stateless/testable and means concurrent
sessions/users never share state by accident, since each caller's loop
holds its own instance.

Lifetime: exactly one conversation/session, however long the caller
chooses to keep that loop alive. There is no automatic expiry - state
persists until explicitly replaced by a new resolution, since the whole
point is that a user can go quiet for a while and still have "what about
EPS?" resolve against the last thing they were discussing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """The last successfully resolved query in this conversation, if any."""

    symbol: str | None = None
    line_items: list[str] = field(default_factory=list)
    derived_metrics: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    needs_qualitative_context: bool = False

    def is_empty(self) -> bool:
        return self.symbol is None

    def as_prompt_context(self) -> str:
        """Render as a short text block to prepend to the Query
        Understanding prompt, so the LLM can use it as fallback context
        for an underspecified follow-up question.
        """
        if self.is_empty():
            return "(No prior context in this conversation - this is the first question.)"

        parts = [f"Company: {self.symbol}"]
        if self.line_items:
            parts.append(f"Line items previously discussed: {', '.join(self.line_items)}")
        if self.derived_metrics:
            parts.append(f"Derived metrics previously discussed: {', '.join(self.derived_metrics)}")
        if self.periods:
            parts.append(f"Periods previously discussed: {', '.join(self.periods)}")
        return "Prior context from this conversation (use ONLY if the new question " \
               "doesn't specify its own company/period - a new company mentioned " \
               "in the question always overrides this):\n" + "\n".join(parts)

    @classmethod
    def from_resolved_query(cls, resolved) -> "ConversationState":
        """Build the next state from a ResolvedQuery that successfully
        proceeded (can_proceed=True). Callers should NOT update state from
        a stopped/clarification-only turn, since there's nothing new to
        remember in that case.
        """
        return cls(
            symbol=resolved.symbol,
            line_items=list(resolved.line_items),
            derived_metrics=list(resolved.derived_metrics),
            periods=list(resolved.periods),
            needs_qualitative_context=resolved.needs_qualitative_context,
        )
