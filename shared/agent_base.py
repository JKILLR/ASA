"""Base agent configuration and wrapper for Claude Agent SDK."""

from dataclasses import dataclass, field
from typing import Any

from claude_code_sdk import Agent, Message, query


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""

    name: str
    role: str = "worker"
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    max_turns: int = 10
    settings: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base agent class that wraps the Claude Agent SDK."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._history: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> str:
        return self.config.role

    async def run(self, prompt: str, **kwargs: Any) -> str:
        """Execute the agent with the given prompt."""
        system = self.config.system_prompt or self._default_system_prompt()

        messages: list[Message] = []

        async for event in query(
            prompt=prompt,
            system=system,
            model=self.config.model,
            max_turns=kwargs.get("max_turns", self.config.max_turns),
        ):
            if isinstance(event, Message):
                messages.append(event)

        # Extract text content from messages
        result_parts = []
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        result_parts.append(block.text)

        result = "\n".join(result_parts)
        self._history.append({"prompt": prompt, "response": result})
        return result

    def _default_system_prompt(self) -> str:
        """Generate a default system prompt based on role."""
        role_prompts = {
            "orchestrator": (
                f"You are {self.name}, an orchestrator agent. "
                "Coordinate tasks, delegate work, and ensure quality outcomes."
            ),
            "worker": (
                f"You are {self.name}, a worker agent. "
                "Execute assigned tasks efficiently and report progress."
            ),
            "critic": (
                f"You are {self.name}, a critic agent. "
                "Review work, provide feedback, and suggest improvements."
            ),
        }
        return role_prompts.get(
            self.role,
            f"You are {self.name}, an AI agent with role: {self.role}."
        )

    def get_history(self) -> list[dict[str, str]]:
        """Get the conversation history for this agent."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history = []
