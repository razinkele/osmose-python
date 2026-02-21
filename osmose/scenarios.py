"""Scenario management for OSMOSE configurations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Scenario:
    """A named, versioned OSMOSE configuration snapshot."""
    name: str
    description: str = ""
    created_at: str = ""  # ISO format
    modified_at: str = ""  # ISO format
    config: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_scenario: str | None = None

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.modified_at:
            self.modified_at = now


@dataclass
class ParamDiff:
    """A single parameter difference between two scenarios."""
    key: str
    value_a: str | None
    value_b: str | None


class ScenarioManager:
    """Save, load, compare, and fork OSMOSE scenarios."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, scenario: Scenario) -> Path:
        """Save a scenario to disk as JSON."""
        scenario.modified_at = datetime.now().isoformat()
        scenario_dir = self.storage_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(scenario)
        with open(scenario_dir / "scenario.json", "w") as f:
            json.dump(data, f, indent=2)
        return scenario_dir

    def load(self, name: str) -> Scenario:
        """Load a named scenario from disk."""
        path = self.storage_dir / name / "scenario.json"
        with open(path) as f:
            data = json.load(f)
        return Scenario(**data)

    def list_scenarios(self) -> list[dict[str, str]]:
        """List all saved scenarios with basic metadata."""
        results = []
        for d in sorted(self.storage_dir.iterdir()):
            json_path = d / "scenario.json"
            if d.is_dir() and json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                results.append({
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "modified_at": data.get("modified_at", ""),
                    "tags": data.get("tags", []),
                })
        return results

    def delete(self, name: str) -> None:
        """Delete a saved scenario."""
        path = self.storage_dir / name
        if path.exists():
            shutil.rmtree(path)

    def compare(self, name_a: str, name_b: str) -> list[ParamDiff]:
        """Compare two scenarios and return parameter differences."""
        a = self.load(name_a)
        b = self.load(name_b)
        all_keys = sorted(set(a.config.keys()) | set(b.config.keys()))
        diffs = []
        for key in all_keys:
            val_a = a.config.get(key)
            val_b = b.config.get(key)
            if val_a != val_b:
                diffs.append(ParamDiff(key=key, value_a=val_a, value_b=val_b))
        return diffs

    def fork(self, source_name: str, new_name: str, description: str = "") -> Scenario:
        """Create a new scenario based on an existing one."""
        source = self.load(source_name)
        forked = Scenario(
            name=new_name,
            description=description or f"Forked from {source_name}",
            config=dict(source.config),
            tags=list(source.tags),
            parent_scenario=source_name,
        )
        self.save(forked)
        return forked
