"""Swarm interface and configuration for the hierarchical agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SwarmConfig:
    """Configuration for a swarm, parsed from swarm.yaml."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    agents: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "SwarmConfig":
        """Load swarm configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            name=data.get("name", path.parent.name),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            agents=data.get("agents", []),
            settings=data.get("settings", {}),
        )


class SwarmInterface(ABC):
    """Base interface for all swarms."""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self._agents: dict[str, Any] = {}

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get current status of the swarm."""
        pass

    @abstractmethod
    def get_priorities(self) -> list[str]:
        """Get current priority tasks for the swarm."""
        pass

    @abstractmethod
    def receive_directive(self, directive: str) -> str:
        """Receive and process a directive from the orchestrator."""
        pass

    @abstractmethod
    def report_progress(self) -> dict[str, Any]:
        """Report progress on current tasks."""
        pass


class Swarm(SwarmInterface):
    """Default swarm implementation that loads from swarm.yaml and manages agents."""

    def __init__(self, swarm_path: Path):
        self.swarm_path = swarm_path
        config_path = swarm_path / "swarm.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"No swarm.yaml found at {config_path}")

        config = SwarmConfig.from_yaml(config_path)
        super().__init__(config)

        self._status = "idle"
        self._priorities: list[str] = []
        self._progress: dict[str, Any] = {}

    def get_status(self) -> dict[str, Any]:
        """Get current status of the swarm."""
        return {
            "name": self.config.name,
            "status": self._status,
            "agent_count": len(self.config.agents),
            "agents": [a.get("name", "unnamed") for a in self.config.agents],
        }

    def get_priorities(self) -> list[str]:
        """Get current priority tasks for the swarm."""
        return self._priorities.copy()

    def set_priorities(self, priorities: list[str]) -> None:
        """Set priority tasks for the swarm."""
        self._priorities = priorities

    def receive_directive(self, directive: str) -> str:
        """Receive and process a directive from the orchestrator."""
        self._status = "working"
        self._priorities.insert(0, directive)
        return f"Swarm '{self.config.name}' received directive: {directive}"

    def report_progress(self) -> dict[str, Any]:
        """Report progress on current tasks."""
        return {
            "swarm": self.config.name,
            "status": self._status,
            "current_priorities": self._priorities,
            "progress": self._progress,
        }

    def initialize_agents(self) -> None:
        """Initialize agents defined in the swarm configuration."""
        from .agent_base import AgentConfig, BaseAgent

        for agent_def in self.config.agents:
            agent_config = AgentConfig(
                name=agent_def.get("name", "unnamed"),
                role=agent_def.get("role", "worker"),
                model=agent_def.get("model", "claude-sonnet-4-20250514"),
                system_prompt=agent_def.get("system_prompt", ""),
            )
            self._agents[agent_config.name] = BaseAgent(agent_config)

    def get_agent(self, name: str) -> Any:
        """Get an agent by name."""
        return self._agents.get(name)
