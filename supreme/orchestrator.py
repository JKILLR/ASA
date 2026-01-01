"""Supreme Orchestrator - Top-level coordinator for all swarms."""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.swarm_interface import Swarm, SwarmConfig


class SupremeOrchestrator:
    """Top-level orchestrator that manages all swarms."""

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.swarms_path = self.base_path / "swarms"
        self._swarms: dict[str, Swarm] = {}
        self._discover_swarms()

    def _discover_swarms(self) -> None:
        """Auto-discover swarms in the swarms/ directory."""
        if not self.swarms_path.exists():
            return

        for item in self.swarms_path.iterdir():
            # Skip template and hidden directories
            if item.name.startswith("_") or item.name.startswith("."):
                continue

            if item.is_dir():
                swarm_yaml = item / "swarm.yaml"
                if swarm_yaml.exists():
                    try:
                        swarm = Swarm(item)
                        self._swarms[swarm.config.name] = swarm
                    except Exception as e:
                        print(f"Warning: Failed to load swarm at {item}: {e}")

    def list_swarms(self) -> list[dict[str, Any]]:
        """List all discovered swarms with their status."""
        return [swarm.get_status() for swarm in self._swarms.values()]

    def get_swarm(self, name: str) -> Swarm | None:
        """Get a swarm by name."""
        return self._swarms.get(name)

    def route_request(self, request: str, swarm_name: str | None = None) -> str:
        """Route a request to the appropriate swarm."""
        if not self._swarms:
            return "No swarms available. Create a swarm first."

        # If swarm specified, use it directly
        if swarm_name:
            swarm = self._swarms.get(swarm_name)
            if not swarm:
                return f"Swarm '{swarm_name}' not found."
            return swarm.receive_directive(request)

        # Auto-route to first available swarm (simple heuristic for now)
        # In production, this could use semantic matching
        swarm = next(iter(self._swarms.values()))
        return swarm.receive_directive(request)

    def get_all_status(self) -> dict[str, Any]:
        """Get status of all swarms."""
        return {
            "swarm_count": len(self._swarms),
            "swarms": {name: swarm.get_status() for name, swarm in self._swarms.items()},
        }

    def get_all_progress(self) -> dict[str, Any]:
        """Get progress reports from all swarms."""
        return {
            name: swarm.report_progress()
            for name, swarm in self._swarms.items()
        }


def main() -> None:
    """CLI interface for the Supreme Orchestrator."""
    parser = argparse.ArgumentParser(
        description="Supreme Orchestrator - Manage hierarchical agent swarms"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all swarms")

    # Status command
    subparsers.add_parser("status", help="Get status of all swarms")

    # Progress command
    subparsers.add_parser("progress", help="Get progress from all swarms")

    # Route command
    route_parser = subparsers.add_parser("route", help="Route a request to a swarm")
    route_parser.add_argument("request", help="The request to route")
    route_parser.add_argument("--swarm", "-s", help="Target swarm name")

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive mode")

    args = parser.parse_args()

    orchestrator = SupremeOrchestrator()

    if args.command == "list":
        swarms = orchestrator.list_swarms()
        if not swarms:
            print("No swarms discovered.")
        else:
            print("Discovered Swarms:")
            for swarm in swarms:
                print(f"  - {swarm['name']}: {swarm['status']} ({swarm['agent_count']} agents)")

    elif args.command == "status":
        status = orchestrator.get_all_status()
        print(f"Total Swarms: {status['swarm_count']}")
        for name, info in status["swarms"].items():
            print(f"\n{name}:")
            print(f"  Status: {info['status']}")
            print(f"  Agents: {', '.join(info['agents'])}")

    elif args.command == "progress":
        progress = orchestrator.get_all_progress()
        for name, report in progress.items():
            print(f"\n{name}:")
            print(f"  Status: {report['status']}")
            print(f"  Priorities: {report['current_priorities']}")

    elif args.command == "route":
        result = orchestrator.route_request(args.request, args.swarm)
        print(result)

    elif args.command == "interactive":
        print("Supreme Orchestrator - Interactive Mode")
        print("Commands: list, status, progress, route <request>, quit")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            if user_input == "quit":
                break
            elif user_input == "list":
                swarms = orchestrator.list_swarms()
                if not swarms:
                    print("No swarms discovered.")
                else:
                    for swarm in swarms:
                        print(f"  - {swarm['name']}: {swarm['status']}")
            elif user_input == "status":
                status = orchestrator.get_all_status()
                print(f"Total: {status['swarm_count']} swarms")
            elif user_input == "progress":
                progress = orchestrator.get_all_progress()
                for name, report in progress.items():
                    print(f"  {name}: {report['status']}")
            elif user_input.startswith("route "):
                request = user_input[6:]
                result = orchestrator.route_request(request)
                print(result)
            else:
                print(f"Unknown command: {user_input}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
